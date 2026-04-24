from __future__ import annotations

import logging
from typing import Dict, List

import torch

logger = logging.getLogger("federated_qa")


class FederatedServer:
    """
    Aggregates GNN parameters from all clients using weighted FedAvg.

    Only the GNN parameters (φ) are federated; LoRA and FiLM weights
    remain client-local.
    """

    def __init__(self) -> None:
        self._global_state: Dict[str, torch.Tensor] = {}

    def aggregate(
        self,
        client_states: List[Dict[str, torch.Tensor]],
        weights: List[float],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute weighted FedAvg of client GNN state dicts.

        Args:
            client_states: list of GNN state_dict() from each client (on CPU)
            weights: n_k training-sample counts per client

        Returns:
            global_state: averaged state dict (on CPU)
        """
        assert len(client_states) == len(weights), "State/weight count mismatch"
        total = sum(weights)
        norm_w = [w / total for w in weights]

        global_state: Dict[str, torch.Tensor] = {}
        for key in client_states[0].keys():
            aggregated = sum(
                norm_w[i] * client_states[i][key].float()
                for i in range(len(client_states))
            )
            global_state[key] = aggregated.to(dtype=client_states[0][key].dtype)

        self._global_state = global_state

        # Log norm of global GNN parameter vector
        total_norm = sum(
            v.float().norm().item() ** 2 for v in global_state.values()
        ) ** 0.5
        logger.info(f"Server aggregation complete | φ_global L2-norm = {total_norm:.4f}")

        return global_state

    def get_global_state(self) -> Dict[str, torch.Tensor]:
        """Return the most recent global GNN state dict."""
        return self._global_state

    def global_param_norm(self) -> float:
        """Return the L2 norm of the current global GNN parameters."""
        if not self._global_state:
            return 0.0
        return sum(
            v.float().norm().item() ** 2 for v in self._global_state.values()
        ) ** 0.5
