from __future__ import annotations

from dataclasses import dataclass

import torch

from .schema import PerceptionBatch

#代替真实数据，测试接口的匹配与数据流动

PADDING_ROLE_ID = 0
ROBOT_ROLE_ID = 1
OBJECT_ROLE_ID = 2


@dataclass(frozen=True)
class MockPerceptionConfig:
    batch_size: int = 2
    history_frames: int = 4  #历史帧数
    object_slots: int = 16  #最多有几个物体slot
    valid_object_counts: tuple[int, ...] | None = None  #每批次当中各样本实际有几个物体
    text_t1_len: int = 4
    text_t2_len: int = 12
    vq_tokens_per_frame: int = 256  #每帧视觉token数目
    dino_dim: int = 1024
    mask_height: int = 8
    mask_width: int = 8
    seed: int = 0
    dtype: torch.dtype = torch.float32
    device: torch.device | str = "cpu"

    @property
    def slot_count(self) -> int:
        return self.object_slots + 1  #将slot0预留给机器人自身


class MockPerceptionCache:
    """Deterministic mock perception for M1-M6 shape and invariant tests."""

    def __init__(self, config: MockPerceptionConfig | None = None) -> None:
        self.config = config or MockPerceptionConfig()

    def generate(self) -> PerceptionBatch:
        cfg = self.config
        device = torch.device(cfg.device)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(cfg.seed)

        bsz = cfg.batch_size
        frames = cfg.history_frames
        slots = cfg.slot_count

        valid_object_counts = self._valid_object_counts(cfg)
        slot_valid_mask = torch.zeros((bsz, frames, slots), dtype=torch.bool)  #该slot是否真实存在
        slot_is_object = torch.zeros((bsz, frames, slots), dtype=torch.bool)  #该slot是否是物体
        slot_role_ids = torch.full((bsz, frames, slots), PADDING_ROLE_ID, dtype=torch.long)  #这个slot是padding/机器人/物体
        object_track_ids = torch.full((bsz, frames, slots), -1, dtype=torch.long)  #跨帧追踪id

        slot_valid_mask[:, :, 0] = True #在所有帧中机器人都存在
        slot_role_ids[:, :, 0] = ROBOT_ROLE_ID
        object_track_ids[:, :, 0] = 0

        target_slot_ids = torch.ones((bsz,), dtype=torch.long)

        #按照每个样本实际物体数填充物体slot
        for batch_idx, object_count in enumerate(valid_object_counts):
            #valid_object_counts指的是batch中真实物体的个数如(2,0)表示batch0有2个物体，batch1没有物体
            if object_count > 0:
                object_slice = slice(1, object_count + 1)
                slot_valid_mask[batch_idx, :, object_slice] = True #object_slice这些slot是有效的
                slot_is_object[batch_idx, :, object_slice] = True
                slot_role_ids[batch_idx, :, object_slice] = OBJECT_ROLE_ID
                track_ids = torch.arange(1, object_count + 1, dtype=torch.long) 
                object_track_ids[batch_idx, :, object_slice] = track_ids.view(1, -1) #跨帧追踪的idea，赋值以后每一帧都是一样的
                target_slot_ids[batch_idx] = 1
            else:
                target_slot_ids[batch_idx] = 0

        addr_t0 = torch.randn((bsz, slots, 32), generator=generator, dtype=cfg.dtype)  #随机生成32维addr，注意这里没有T，身份向量应该在帧之间保持不变，只在第一帧生成一次
        addr_t0 = addr_t0 * slot_valid_mask[:, 0, :].unsqueeze(-1).to(cfg.dtype)
        addr = addr_t0[:, None, :, :].expand(bsz, frames, slots, 32).clone()

        content = torch.randn((bsz, frames, slots, 256), generator=generator, dtype=cfg.dtype)
        pose = torch.randn((bsz, frames, slots, 9), generator=generator, dtype=cfg.dtype)
        dino_features = torch.randn(
            (bsz, frames, slots, cfg.dino_dim), generator=generator, dtype=cfg.dtype
        )
        valid_float = slot_valid_mask.unsqueeze(-1).to(cfg.dtype)
        content = content * valid_float
        pose = pose * valid_float
        dino_features = dino_features * valid_float


        #与生成slot嵌入
        slot_time_emb = _sinusoidal_time_embedding(frames, slots, cfg.dtype)
        slot_time_emb = slot_time_emb.unsqueeze(0).expand(bsz, frames, slots, 16).clone()
        slot_time_emb = slot_time_emb * valid_float
        slot_role_emb = _role_embedding(slot_role_ids).to(dtype=cfg.dtype)
        slot_role_emb = slot_role_emb * valid_float

        text_t1_noun_ids = torch.randint(
            100, 1000, (bsz, cfg.text_t1_len), generator=generator, dtype=torch.long
        )
        text_t2_bpe_ids = torch.randint(
            100, 1000, (bsz, cfg.text_t2_len), generator=generator, dtype=torch.long
        )
        vq_input_ids = torch.randint(
            3,
            8195,
            (bsz, frames, cfg.vq_tokens_per_frame),
            generator=generator,
            dtype=torch.long,
        )
        state_input_ids = torch.randint(
            15500, 16001, (bsz, frames, 7), generator=generator, dtype=torch.long
        )
        action_input_ids = torch.randint(
            10000, 15005, (bsz, max(frames - 1, 0), 7), generator=generator, dtype=torch.long
        )
        sam_masks = torch.rand(
            (bsz, frames, slots, cfg.mask_height, cfg.mask_width), generator=generator
        ) > 0.5
        sam_masks = sam_masks & slot_valid_mask[:, :, :, None, None]

        return PerceptionBatch(
            text_t1_noun_ids=text_t1_noun_ids.to(device),
            text_t2_bpe_ids=text_t2_bpe_ids.to(device),
            vq_input_ids=vq_input_ids.to(device),
            state_input_ids=state_input_ids.to(device),
            action_input_ids=action_input_ids.to(device),
            addr=addr.to(device),
            content=content.to(device),
            pose=pose.to(device),
            slot_valid_mask=slot_valid_mask.to(device),
            slot_is_object=slot_is_object.to(device),
            slot_role_ids=slot_role_ids.to(device),
            slot_time_emb=slot_time_emb.to(device),
            slot_role_emb=slot_role_emb.to(device),
            dino_features=dino_features.to(device),
            sam_masks=sam_masks.to(device),
            object_track_ids=object_track_ids.to(device),
            target_slot_ids=target_slot_ids.to(device),
        )

    @staticmethod
    def _valid_object_counts(config: MockPerceptionConfig) -> tuple[int, ...]:
        if config.valid_object_counts is None:
            default_count = max(config.object_slots - 1, 0)
            return tuple(default_count for _ in range(config.batch_size))
        if len(config.valid_object_counts) != config.batch_size:
            raise ValueError("valid_object_counts must match batch_size")
        for count in config.valid_object_counts:
            if count < 0 or count > config.object_slots:
                raise ValueError("valid object count must be in [0, object_slots]")
        return config.valid_object_counts


def _sinusoidal_time_embedding(frames: int, slots: int, dtype: torch.dtype) -> torch.Tensor:
    positions = torch.arange(frames, dtype=torch.float32).unsqueeze(1)
    dims = torch.arange(0, 16, 2, dtype=torch.float32)
    div_term = torch.exp(-torch.log(torch.tensor(10000.0)) * dims / 16)
    emb = torch.zeros((frames, 16), dtype=torch.float32)
    emb[:, 0::2] = torch.sin(positions * div_term)
    emb[:, 1::2] = torch.cos(positions * div_term)
    return emb[:, None, :].expand(frames, slots, 16).to(dtype=dtype)


def _role_embedding(slot_role_ids: torch.Tensor) -> torch.Tensor:
    table = torch.zeros((3, 16), dtype=torch.float32)
    table[ROBOT_ROLE_ID, 0] = 1.0
    table[OBJECT_ROLE_ID, 1] = 1.0
    return table[slot_role_ids]

