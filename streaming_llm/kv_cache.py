import torch
import random

def slice2d(x, start, end):
    return x[:, :, start:end, ...]


def slice3d(x, start, end):
    return x[:, :, :, start:end, ...]


def slice1d(x, start, end):
    return x[:, start:end, ...]


DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
}


class StartRecentKVCache:
    def __init__(
        self,
        start_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
    ):
        print(f"StartRecentKVCache: {start_size}, {recent_size}")
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = start_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]

    def __call__(self, past_key_values):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size),
                        self.k_slice(k, seq_len - self.recent_size, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size),
                        self.v_slice(v, seq_len - self.recent_size, seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]

    def evict_for_space(self, past_key_values, num_coming):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len + num_coming <= self.cache_size:
            return past_key_values
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size),
                        self.k_slice(
                            k, seq_len - self.recent_size + num_coming, seq_len
                        ),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size),
                        self.v_slice(
                            v, seq_len - self.recent_size + num_coming, seq_len
                        ),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]

    def evict_range(self, past_key_values, start, end):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        assert start <= end and end <= seq_len
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, start),
                        self.k_slice(k, end, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, start),
                        self.v_slice(v, end, seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]

class Punctuation_Cache:
    def __init__(self, punc_size=4, k_seq_dim=2, v_seq_dim=2):
        print(f"Punctuation_Cache: {punc_size}")
        self.punc_size = punc_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]
    
    # Ordinary call
    def __call__(self, past_puncs):
        if past_puncs is None:
            return None
        seq_len = past_puncs[0][0].size(self.k_seq_dim)
        if seq_len <= self.punc_size:
            return past_puncs
        return [
            [
            torch.cat(
                [
                    self.k_slice(k, seq_len - self.punc_size, seq_len),
                ],
                dim=self.k_seq_dim,
            ), 
            torch.cat(
                [
                    self.v_slice(v, seq_len - self.punc_size, seq_len),
                ],
                dim=self.v_seq_dim,
            )
            ]for k, v in past_puncs
        ]

    ## Replace the saved puncs with randomly generated tensors
    # def __call__(self, past_puncs):
    #     if past_puncs is None:
    #         return None
    #     seq_len = past_puncs[0][0].size(self.k_seq_dim)
    #     if seq_len <= self.punc_size:
    #         return [[torch.zeros_like(k), torch.zeros_like(v)] for k, v in past_puncs]
    #     return [
    #         [
    #         torch.cat(
    #             [
    #                 self.k_slice(torch.zeros_like(k), seq_len - self.punc_size, seq_len),
    #             ],
    #             dim=self.k_seq_dim,
    #         ), 
    #         torch.cat(
    #             [
    #                 self.v_slice(torch.zeros_like(v), seq_len - self.punc_size, seq_len),
    #             ],
    #             dim=self.v_seq_dim,
    #         )
    #         ]for k, v in past_puncs
    #     ]
  


class StartRecentKVCache_Punc:
    def __init__(
        self,
        start_size=4,
        punc_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
    ):
        print(f"StartRecentKVCache: {start_size}, {punc_size}, {recent_size}")
        self.start_size = start_size
        self.recent_size = recent_size
        self.punc_size = punc_size
        self.cache_size = start_size + recent_size + punc_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]

    def __call__(self, past_key_values, past_puncs):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values
        
        if past_puncs:

            ## Randomly replace past_puncs with random tensors
            # past_puncs = [[torch.randn_like(k) if random.random() < 0.5 else k, torch.randn_like(v) if random.random() < 0.5 else v] for k, v in past_puncs]

            return [
                [
                    torch.cat(
                        [
                            self.k_slice(k, 0, self.start_size),
                            past_puncs[i][0],
                            # self.k_slice(k, seq_len - self.punc_size, seq_len), ## Randomly replace past_puncs with tensors from past_key_values
                            self.k_slice(k, seq_len - self.recent_size, seq_len),
                        ],
                    dim=self.k_seq_dim,
                ), 
                    torch.cat(
                        [
                            self.v_slice(v, 0, self.start_size),
                            past_puncs[i][1],
                            # self.v_slice(v, seq_len - self.punc_size, seq_len), ## Randomly replace past_puncs with tensors from past_key_values
                            self.v_slice(v, seq_len - self.recent_size, seq_len),
                        ],
                    dim=self.v_seq_dim,
                ),
                ]
                for i, (k, v) in enumerate(past_key_values)
            ]
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size),
                        self.k_slice(k, seq_len - self.recent_size, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size),
                        self.v_slice(v, seq_len - self.recent_size, seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]

    def evict_for_space(self, past_key_values, num_coming):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len + num_coming <= self.cache_size:
            return past_key_values
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size),
                        self.k_slice(
                            k, seq_len - self.recent_size + num_coming, seq_len
                        ),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size),
                        self.v_slice(
                            v, seq_len - self.recent_size + num_coming, seq_len
                        ),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]
