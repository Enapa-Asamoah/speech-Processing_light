"""
Utility helpers for weight pruning.

Supports unstructured L1 pruning and simple structured pruning for Conv/Linear
layers, plus reporting and cleanup to make pruned weights permanent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


@dataclass
class PruningSpec:
	"""Configuration for pruning a model."""

	amount: float = 0.3  # fraction of weights to prune
	structured: bool = False  # use LN structured pruning when True
	norm: int = 2  # only used when structured=True
	bias: bool = False  # also prune bias terms


def _default_prunable_modules(model: nn.Module) -> List[Tuple[nn.Module, str]]:
	"""Collect parameters eligible for pruning (weights, optional biases)."""

	candidates: List[Tuple[nn.Module, str]] = []
	prunable_types = (nn.Conv1d, nn.Conv2d, nn.Linear)
	for module in model.modules():
		if isinstance(module, prunable_types):
			if hasattr(module, "weight"):
				candidates.append((module, "weight"))
			if hasattr(module, "bias") and module.bias is not None:
				candidates.append((module, "bias"))
	return candidates


def apply_pruning(model: nn.Module, spec: PruningSpec) -> None:
	"""Apply pruning in-place on the model according to the spec."""

	params_to_prune = _default_prunable_modules(model)
	if not spec.bias:
		params_to_prune = [(m, n) for (m, n) in params_to_prune if n == "weight"]

	if not params_to_prune:
		print("No prunable parameters found.")
		return

	if spec.structured:
		# Structured LN pruning removes entire channels/filters.
		for module, name in params_to_prune:
			prune.ln_structured(module, name=name, amount=spec.amount, n=spec.norm, dim=0)
	else:
		# Global unstructured L1 pruning across all selected parameters.
		prune.global_unstructured(
			params_to_prune,
			pruning_method=prune.L1Unstructured,
			amount=spec.amount,
		)


def remove_reparametrization(model: nn.Module) -> None:
	"""Remove pruning reparametrization so masks become permanent weights."""

	for module in model.modules():
		for name in ("weight", "bias"):
			if prune.is_pruned(module):
				try:
					prune.remove(module, name)
				except ValueError:
					# skip if this param was not pruned
					continue


def report_sparsity(model: nn.Module) -> Tuple[float, List[Tuple[str, float]]]:
	"""Return overall sparsity and per-layer sparsity summary."""

	per_layer: List[Tuple[str, float]] = []
	total_zeros, total_params = 0, 0

	for name, module in model.named_modules():
		if not hasattr(module, "weight"):
			continue
		weight = module.weight.detach()
		zeros = torch.sum(weight == 0).item()
		params = weight.numel()
		if params == 0:
			continue
		sparsity = zeros / params
		per_layer.append((name, sparsity))
		total_zeros += zeros
		total_params += params

	overall = (total_zeros / total_params) if total_params > 0 else 0.0
	return overall, per_layer


def prune_and_report(model: nn.Module, spec: PruningSpec) -> Tuple[float, List[Tuple[str, float]]]:
	"""Apply pruning, remove reparametrization, and return sparsity metrics."""

	apply_pruning(model, spec)
	remove_reparametrization(model)
	return report_sparsity(model)


def fine_tune_after_pruning(
	model: nn.Module,
	train_loader: torch.utils.data.DataLoader,
	val_loader: torch.utils.data.DataLoader,
	epochs: int = 5,
	lr: float = 1e-4,
	device: torch.device | None = None,
	verbose: bool = True,
) -> Tuple[float, float]:
	"""Lightweight fine-tuning loop to recover accuracy after pruning."""

	device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
	model = model.to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	for epoch in range(1, epochs + 1):
		model.train()
		running_loss = 0.0
		for xb, yb in train_loader:
			xb, yb = xb.to(device), yb.to(device)
			optimizer.zero_grad()
			logits = model(xb)
			loss = criterion(logits, yb)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
		if verbose:
			avg_loss = running_loss / max(1, len(train_loader))
			print(f"[FT] Epoch {epoch}/{epochs} loss={avg_loss:.4f}")

	# quick eval
	model.eval()
	correct, total = 0, 0
	val_loss = 0.0
	with torch.no_grad():
		for xb, yb in val_loader:
			xb, yb = xb.to(device), yb.to(device)
			logits = model(xb)
			val_loss += criterion(logits, yb).item()
			preds = torch.argmax(logits, dim=1)
			correct += (preds == yb).sum().item()
			total += yb.numel()

	val_loss /= max(1, len(val_loader))
	val_acc = correct / total if total else 0.0
	return val_loss, val_acc
