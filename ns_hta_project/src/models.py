"""
models.py - Neuro-Symbolic Hyper-Tensor Alignment (NS-HTA) architecture and ablations.

Implements:
1. NS-HTA Pairwise Model:
   - Text projection: h_T = ReLU(W_T x_T + b_T) in R^32
   - Symbolic projection: h_S = ReLU(W_S x_S + b_S) in R^32
   - Hadamard interaction: h_TS = h_T (element-wise mul) h_S in R^32
   - Fused latent vector: z_TS = [h_T; h_S; h_TS] in R^96
   - Classification: p_hat = softmax(W_C z_TS + b_C) in R^4
2. Ablated Pairwise Model (without interaction):
   - z_0 = [h_T; h_S] in R^64
3. Concatenation MLP baseline:
   - Input: [x_T; x_S] in R^389 -> Hidden 64 -> Output 4
4. Representation-Level Attribution (Eq. 14):
   - Computes mean absolute weight magnitude A_k for text, symbolic, and interaction blocks.
   - Computes activation-normalized attribution to address scale disparity across blocks.
5. Exact parameter count inspection (exact 12,900 weights for NS-HTA).
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


class NSHTAPairwiseModel:
    """
    Empirical NS-HTA architecture implementing Equations (6)-(14).
    Pure NumPy implementation for robust execution and exact parameter control.
    """
    def __init__(
        self,
        d_text: int = 384,
        d_sym: int = 5,
        d_latent: int = 32,
        num_classes: int = 4,
        with_interaction: bool = True,
        learning_rate: float = 0.01,
        weight_decay: float = 1e-4,
        random_seed: int = 42,
        use_class_weights: bool = False
    ):
        self.d_text = d_text
        self.d_sym = d_sym
        self.d_latent = d_latent
        self.num_classes = num_classes
        self.with_interaction = with_interaction
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.use_class_weights = use_class_weights
        self.rng = np.random.RandomState(random_seed)
        
        # Fused dimension: 32 + 32 + 32 = 96 if with_interaction, else 32 + 32 = 64
        self.d_fused = (d_latent * 3) if with_interaction else (d_latent * 2)
        
        # Initialize learnable parameters using He/Xavier uniform initialization
        # Text projection: W_T (32, 384), b_T (32) -> 12,320 params
        limit_T = np.sqrt(6.0 / (d_text + d_latent))
        self.W_T = self.rng.uniform(-limit_T, limit_T, (d_latent, d_text)).astype(np.float32)
        self.b_T = np.zeros(d_latent, dtype=np.float32)
        
        # Symbolic projection: W_S (32, 5), b_S (32) -> 192 params
        limit_S = np.sqrt(6.0 / (d_sym + d_latent))
        self.W_S = self.rng.uniform(-limit_S, limit_S, (d_latent, d_sym)).astype(np.float32)
        self.b_S = np.zeros(d_latent, dtype=np.float32)
        
        # Classification head: W_C (4, 96 or 64), b_C (4) -> 388 or 260 params
        limit_C = np.sqrt(6.0 / (self.d_fused + num_classes))
        self.W_C = self.rng.uniform(-limit_C, limit_C, (num_classes, self.d_fused)).astype(np.float32)
        self.b_C = np.zeros(num_classes, dtype=np.float32)

    def count_parameters(self) -> Dict[str, int]:
        """
        Returns exact trainable parameter breakdown.
        """
        p_text = self.W_T.size + self.b_T.size
        p_sym = self.W_S.size + self.b_S.size
        p_head = self.W_C.size + self.b_C.size
        total = p_text + p_sym + p_head
        return {
            "W_T + b_T (Text Projection)": p_text,
            "W_S + b_S (Symbolic Projection)": p_sym,
            "W_C + b_C (Classification Head)": p_head,
            "Total Trainable Parameters": total
        }

    def forward(self, x_T: np.ndarray, x_S: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Forward pass computing equations (6), (8), (9), (10), (12).
        
        Args:
            x_T: (N, 384) text embeddings
            x_S: (N, 5) binary symbolic features
            
        Returns:
            p_hat: (N, 4) class probabilities
            cache: intermediate activation tensors
        """
        N = x_T.shape[0]
        
        # (6) h_T = ReLU(W_T x_T + b_T)
        z_T_pre = np.dot(x_T, self.W_T.T) + self.b_T
        h_T = relu(z_T_pre)
        
        # (8) h_S = ReLU(W_S x_S + b_S)
        z_S_pre = np.dot(x_S, self.W_S.T) + self.b_S
        h_S = relu(z_S_pre)
        
        if self.with_interaction:
            # (9) h_TS = h_T (hadamard) h_S
            h_TS = h_T * h_S
            # (10) z_TS = [h_T; h_S; h_TS] in R^96
            z_fused = np.concatenate([h_T, h_S, h_TS], axis=1)
        else:
            # (11) z_0 = [h_T; h_S] in R^64
            h_TS = None
            z_fused = np.concatenate([h_T, h_S], axis=1)
            
        # (12) p_hat = softmax(W_C z_fused + b_C)
        logits = np.dot(z_fused, self.W_C.T) + self.b_C
        p_hat = softmax(logits)
        
        cache = {
            "x_T": x_T, "x_S": x_S,
            "z_T_pre": z_T_pre, "h_T": h_T,
            "z_S_pre": z_S_pre, "h_S": h_S,
            "h_TS": h_TS, "z_fused": z_fused,
            "logits": logits, "p_hat": p_hat
        }
        return p_hat, cache

    def fit(
        self,
        x_T: np.ndarray,
        x_S: np.ndarray,
        y: np.ndarray,
        epochs: int = 60,
        batch_size: int = 16,
        class_weights: Optional[np.ndarray] = None
    ):
        """
        Trains the model using Mini-batch SGD with Adam optimizer and Cross-Entropy loss.
        """
        N = x_T.shape[0]
        
        # Compute class weights if requested
        if self.use_class_weights and class_weights is None:
            counts = np.bincount(y, minlength=self.num_classes)
            class_weights = N / (self.num_classes * np.maximum(counts, 1).astype(np.float32))
            class_weights = class_weights / np.mean(class_weights)
            
        # Adam moments initialization
        params = [self.W_T, self.b_T, self.W_S, self.b_S, self.W_C, self.b_C]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t = 0
        
        for epoch in range(epochs):
            # Shuffle indices
            indices = np.arange(N)
            self.rng.shuffle(indices)
            
            for start_idx in range(0, N, batch_size):
                t += 1
                end_idx = min(start_idx + batch_size, N)
                batch_idx = indices[start_idx:end_idx]
                B = len(batch_idx)
                
                xb_T = x_T[batch_idx]
                xb_S = x_S[batch_idx]
                yb = y[batch_idx]
                
                p_hat, cache = self.forward(xb_T, xb_S)
                
                # Cross-entropy loss gradient
                one_hot = np.zeros_like(p_hat)
                one_hot[np.arange(B), yb] = 1.0
                
                d_logits = p_hat - one_hot
                if class_weights is not None:
                    weights_b = class_weights[yb][:, None]
                    d_logits = d_logits * weights_b
                d_logits = d_logits / B
                
                # Backprop into W_C and b_C
                z_fused = cache["z_fused"]
                dW_C = np.dot(d_logits.T, z_fused) + self.weight_decay * self.W_C
                db_C = np.sum(d_logits, axis=0)
                
                # Backprop into fused vector
                dz_fused = np.dot(d_logits, self.W_C)
                
                h_T = cache["h_T"]
                h_S = cache["h_S"]
                
                if self.with_interaction:
                    dh_T_direct = dz_fused[:, :self.d_latent]
                    dh_S_direct = dz_fused[:, self.d_latent:self.d_latent*2]
                    dh_TS = dz_fused[:, self.d_latent*2:]
                    
                    # Interaction gradient: h_TS = h_T * h_S
                    dh_T = dh_T_direct + (dh_TS * h_S)
                    dh_S = dh_S_direct + (dh_TS * h_T)
                else:
                    dh_T = dz_fused[:, :self.d_latent]
                    dh_S = dz_fused[:, self.d_latent:]
                    
                # Backprop through ReLU
                dz_T_pre = dh_T * (cache["z_T_pre"] > 0)
                dz_S_pre = dh_S * (cache["z_S_pre"] > 0)
                
                # Gradients for text and symbolic projection layers
                dW_T = np.dot(dz_T_pre.T, xb_T) + self.weight_decay * self.W_T
                db_T = np.sum(dz_T_pre, axis=0)
                
                dW_S = np.dot(dz_S_pre.T, xb_S) + self.weight_decay * self.W_S
                db_S = np.sum(dz_S_pre, axis=0)
                
                grads = [dW_T, db_T, dW_S, db_S, dW_C, db_C]
                
                # Adam update
                for i in range(len(params)):
                    m[i] = beta1 * m[i] + (1 - beta1) * grads[i]
                    v[i] = beta2 * v[i] + (1 - beta2) * (grads[i] ** 2)
                    m_hat = m[i] / (1.0 - beta1 ** t)
                    v_hat = v[i] / (1.0 - beta2 ** t)
                    params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    def predict(self, x_T: np.ndarray, x_S: np.ndarray) -> np.ndarray:
        p_hat, _ = self.forward(x_T, x_S)
        return np.argmax(p_hat, axis=1)

    def predict_proba(self, x_T: np.ndarray, x_S: np.ndarray) -> np.ndarray:
        p_hat, _ = self.forward(x_T, x_S)
        return p_hat

    def compute_representation_attribution(self, x_T: np.ndarray = None, x_S: np.ndarray = None, y: np.ndarray = None) -> Dict[str, Any]:
        """
        Computes representation-level attribution using:
        1. Weight magnitude Ak (Eq. 14)
        2. Activation-scaled attribution (addresses scale disparity across blocks)
        3. Causal Feature Occlusion (measures actual drop in Macro-F1 when zeroing out blocks)
        """
        if not self.with_interaction:
            W_text = self.W_C[:, :self.d_latent]
            W_sym = self.W_C[:, self.d_latent:]
            return {
                "A_text (Raw)": float(np.mean(np.abs(W_text))),
                "A_symbolic (Raw)": float(np.mean(np.abs(W_sym))),
                "A_interaction (Raw)": 0.0
            }
            
        W_text = self.W_C[:, :self.d_latent]
        W_sym = self.W_C[:, self.d_latent:self.d_latent*2]
        W_int = self.W_C[:, self.d_latent*2:]
        
        raw_text = float(np.mean(np.abs(W_text)))
        raw_sym = float(np.mean(np.abs(W_sym)))
        raw_int = float(np.mean(np.abs(W_int)))
        
        res = {
            "A_text (Raw Mean |W|)": raw_text,
            "A_symbolic (Raw Mean |W|)": raw_sym,
            "A_interaction (Raw Mean |W|)": raw_int
        }
        
        # If data is provided, compute activation-scaled attribution
        if x_T is not None and x_S is not None:
            _, cache = self.forward(x_T, x_S)
            mean_act_T = np.mean(cache["h_T"])
            mean_act_S = np.mean(cache["h_S"])
            mean_act_int = np.mean(cache["h_TS"])
            
            res["A_text (Activation-Scaled)"] = float(raw_text * mean_act_T)
            res["A_symbolic (Activation-Scaled)"] = float(raw_sym * mean_act_S)
            res["A_interaction (Activation-Scaled)"] = float(raw_int * mean_act_int)
            
            # Causal Occlusion Testing (Ablation of individual blocks)
            if y is not None:
                from sklearn.metrics import f1_score
                baseline_preds = self.predict(x_T, x_S)
                baseline_f1 = f1_score(y, baseline_preds, average="macro") * 100.0
                
                # Occlude Text Block (zero out h_T)
                z_no_T = np.concatenate([np.zeros_like(cache["h_T"]), cache["h_S"], cache["h_TS"]], axis=1)
                preds_no_T = np.argmax(softmax(np.dot(z_no_T, self.W_C.T) + self.b_C), axis=1)
                drop_T = baseline_f1 - (f1_score(y, preds_no_T, average="macro") * 100.0)
                
                # Occlude Symbolic Block (zero out h_S)
                z_no_S = np.concatenate([cache["h_T"], np.zeros_like(cache["h_S"]), cache["h_TS"]], axis=1)
                preds_no_S = np.argmax(softmax(np.dot(z_no_S, self.W_C.T) + self.b_C), axis=1)
                drop_S = baseline_f1 - (f1_score(y, preds_no_S, average="macro") * 100.0)
                
                # Occlude Interaction Block (zero out h_TS)
                z_no_int = np.concatenate([cache["h_T"], cache["h_S"], np.zeros_like(cache["h_TS"])], axis=1)
                preds_no_int = np.argmax(softmax(np.dot(z_no_int, self.W_C.T) + self.b_C), axis=1)
                drop_int = baseline_f1 - (f1_score(y, preds_no_int, average="macro") * 100.0)
                
                res["Causal_Impact_Delta_MacroF1"] = {
                    "Occlude Text Block (Drop in % F1)": max(0.0, float(drop_T)),
                    "Occlude Symbolic Block (Drop in % F1)": max(0.0, float(drop_S)),
                    "Occlude Interaction Block (Drop in % F1)": max(0.0, float(drop_int))
                }
            
        return res


# PyTorch Module for standard deep learning workflows & IEEE publication artifact
try:
    import torch
    import torch.nn as nn
    
    class NSHTAPyTorch(nn.Module):
        """
        PyTorch implementation of NS-HTA for direct use with PyTorch trainers.
        """
        def __init__(self, d_text=384, d_sym=5, d_latent=32, num_classes=4, with_interaction=True):
            super().__init__()
            self.with_interaction = with_interaction
            self.d_latent = d_latent
            
            # (6) Text latent projection
            self.fc_text = nn.Linear(d_text, d_latent)
            self.relu = nn.ReLU()
            
            # (8) Symbolic latent projection
            self.fc_sym = nn.Linear(d_sym, d_latent)
            
            # (12) Classification head
            in_dim = d_latent * 3 if with_interaction else d_latent * 2
            self.fc_out = nn.Linear(in_dim, num_classes)
            
        def forward(self, x_T, x_S):
            h_T = self.relu(self.fc_text(x_T))
            h_S = self.relu(self.fc_sym(x_S))
            
            if self.with_interaction:
                h_TS = h_T * h_S  # Hadamard product
                z = torch.cat([h_T, h_S, h_TS], dim=1)
            else:
                z = torch.cat([h_T, h_S], dim=1)
                
            logits = self.fc_out(z)
            return logits

except ImportError:
    NSHTAPyTorch = None
