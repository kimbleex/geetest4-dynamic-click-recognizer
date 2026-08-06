"""
Author: Xu JunLiang
Created: 2026-07-27
File: val_geetest4.py

Description: 通过 GeeTest4 动态点选验证码
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_POSITION_NAMES = ("左上", "右上", "左下", "右下")
_DEFAULT_MIN_APPEARANCE = 0.68
_APPEARANCE_HASH_WEIGHT = 0.55
_APPEARANCE_PIXEL_WEIGHT = 0.45
_TOTAL_HASH_WEIGHT = 0.50
_TOTAL_PIXEL_WEIGHT = 0.40
_TOTAL_STRUCTURE_WEIGHT = 0.10


@dataclass(frozen=True)
class _MatchResult:
    index: int
    position: str
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    score: float
    appearance_score: float
    hash_score: float
    pixel_score: float
    component_score: float


class Geetest4Recognizer:
    def __init__(self, min_appearance: float = _DEFAULT_MIN_APPEARANCE) -> None:
        self.min_appearance = min_appearance

    def find_matching_center(
        self,
        target_path: str | Path,
        background_path: str | Path,
    ) -> tuple[int, int] | None:
        target_path = Path(target_path)
        background_path = Path(background_path)
        results = self._recognize(target_path, background_path)
        eligible_results = [
            result
            for result in results
            if result.appearance_score >= self.min_appearance
        ]
        closest = max(results, key=lambda result: result.appearance_score)

        self._print_scores(target_path, background_path, results)
        if not eligible_results:
            print(
                "\n识别结果: 未找到匹配项。"
                f"最接近的是第 {closest.index} 个（{closest.position}），"
                f"外观相似度={closest.appearance_score:.4f}，"
                f"低于门槛 {self.min_appearance:.4f}"
            )
            return None

        best = max(eligible_results, key=lambda result: result.score)
        print(
            f"\n识别结果: 第 {best.index} 个（{best.position}），"
            f"背景图内中心坐标={best.center}"
        )
        return best.center

    def _recognize(
        self,
        target_path: Path,
        background_path: Path,
    ) -> list[_MatchResult]:
        target_image = self._read_grayscale(target_path)
        background = self._read_grayscale(background_path)
        rectangles = self._find_option_rectangles(background)

        target_mask = self._extract_symbol_mask(target_image)
        normalized_target = self._normalize_mask(target_mask)
        target_hash = self._perceptual_hash(normalized_target)

        results: list[_MatchResult] = []
        for index, (x, y, width, height) in enumerate(rectangles, start=1):
            option_image = background[y : y + height, x : x + width]
            option_mask = self._extract_symbol_mask(option_image)
            normalized_option = self._normalize_mask(option_mask)

            current_hash_score = self._hash_similarity(
                target_hash,
                self._perceptual_hash(normalized_option),
            )
            current_pixel_score = self._dice_similarity(
                normalized_target,
                normalized_option,
            )
            current_component_score = self._structural_similarity(
                target_mask,
                option_mask,
            )
            appearance_score = (
                _APPEARANCE_HASH_WEIGHT * current_hash_score
                + _APPEARANCE_PIXEL_WEIGHT * current_pixel_score
            )
            score = (
                _TOTAL_HASH_WEIGHT * current_hash_score
                + _TOTAL_PIXEL_WEIGHT * current_pixel_score
                + _TOTAL_STRUCTURE_WEIGHT * current_component_score
            )
            results.append(
                _MatchResult(
                    index=index,
                    position=_POSITION_NAMES[index - 1],
                    bbox=(x, y, width, height),
                    center=(x + width // 2, y + height // 2),
                    score=score,
                    appearance_score=appearance_score,
                    hash_score=current_hash_score,
                    pixel_score=current_pixel_score,
                    component_score=current_component_score,
                )
            )
        return results

    @staticmethod
    def _read_grayscale(path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"无法读取图片: {path}")
        return image

    @staticmethod
    def _find_option_rectangles(
        background: np.ndarray,
    ) -> list[tuple[int, int, int, int]]:
        _, block_mask = cv2.threshold(
            background,
            250,
            255,
            cv2.THRESH_BINARY_INV,
        )
        contours, _ = cv2.findContours(
            block_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        min_area = background.size * 0.05
        rectangles = [
            cv2.boundingRect(contour)
            for contour in contours
            if cv2.contourArea(contour) >= min_area
        ]
        if len(rectangles) != 4:
            raise ValueError(
                f"期望从背景图中识别出 4 个候选区域，实际识别到 {len(rectangles)} 个"
            )

        middle_y = background.shape[0] / 2
        top = sorted(
            (rect for rect in rectangles if rect[1] < middle_y),
            key=lambda rect: rect[0],
        )
        bottom = sorted(
            (rect for rect in rectangles if rect[1] >= middle_y),
            key=lambda rect: rect[0],
        )
        if len(top) != 2 or len(bottom) != 2:
            raise ValueError("无法将 4 个候选区域排列成 2 x 2 网格")
        return top + bottom

    @staticmethod
    def _extract_symbol_mask(image: np.ndarray) -> np.ndarray:
        _, mask = cv2.threshold(
            image,
            0,
            255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
        )
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        cleaned = np.zeros_like(mask)
        min_area = max(2, round(image.size * 0.00008))
        for label in range(1, component_count):
            if stats[label, cv2.CC_STAT_AREA] >= min_area:
                cleaned[labels == label] = 255

        ys, xs = np.where(cleaned > 0)
        if xs.size == 0:
            raise ValueError("图片中没有识别到图标前景")
        return cleaned[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

    @staticmethod
    def _normalize_mask(
        mask: np.ndarray,
        size: int = 64,
        padding: int = 8,
    ) -> np.ndarray:
        height, width = mask.shape
        scale = min((size - 2 * padding) / width, (size - 2 * padding) / height)
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        resized = cv2.resize(
            mask,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )
        _, resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)

        normalized = np.zeros((size, size), dtype=np.uint8)
        x = (size - resized_width) // 2
        y = (size - resized_height) // 2
        normalized[y : y + resized_height, x : x + resized_width] = resized
        return normalized

    @staticmethod
    def _perceptual_hash(mask: np.ndarray) -> np.ndarray:
        resized = cv2.resize(mask, (32, 32), interpolation=cv2.INTER_AREA)
        frequencies = cv2.dct(resized.astype(np.float32))[:8, :8].reshape(-1)
        frequencies = frequencies[1:]
        return frequencies > np.median(frequencies)

    @staticmethod
    def _hash_similarity(left: np.ndarray, right: np.ndarray) -> float:
        return float(1.0 - np.mean(np.logical_xor(left, right)))

    @staticmethod
    def _dice_similarity(left: np.ndarray, right: np.ndarray) -> float:
        left_foreground = left > 0
        right_foreground = right > 0
        denominator = np.count_nonzero(left_foreground) + np.count_nonzero(
            right_foreground
        )
        if denominator == 0:
            return 0.0
        intersection = np.count_nonzero(left_foreground & right_foreground)
        return float(2 * intersection / denominator)

    @staticmethod
    def _component_features(mask: np.ndarray) -> tuple[int, np.ndarray, float]:
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        components = [
            stats[index]
            for index in range(1, count)
            if stats[index, cv2.CC_STAT_AREA] >= 2
        ]
        if not components:
            return 0, np.zeros(1, dtype=np.float32), 0.0

        heights = np.array(
            sorted((item[3] for item in components), reverse=True),
            dtype=np.float32,
        )
        heights /= heights.max()
        verticality = float(
            np.mean(
                [
                    np.log1p(min(item[3] / max(item[2], 1), 10.0))
                    for item in components
                ]
            )
        )
        return len(components), heights, verticality

    def _structural_similarity(
        self,
        target: np.ndarray,
        option: np.ndarray,
    ) -> float:
        target_count, target_heights, target_verticality = self._component_features(
            target
        )
        option_count, option_heights, option_verticality = self._component_features(
            option
        )
        max_count = max(target_count, option_count, 1)
        count_score = 1.0 - abs(target_count - option_count) / max_count

        signature_size = max(len(target_heights), len(option_heights))
        target_signature = np.pad(
            target_heights,
            (0, signature_size - len(target_heights)),
        )
        option_signature = np.pad(
            option_heights,
            (0, signature_size - len(option_heights)),
        )
        height_score = 1.0 - float(
            np.mean(np.abs(target_signature - option_signature))
        )
        verticality_score = float(
            np.exp(-abs(target_verticality - option_verticality))
        )
        return 0.55 * count_score + 0.30 * height_score + 0.15 * verticality_score

    @staticmethod
    def _print_scores(
        target_path: Path,
        background_path: Path,
        results: list[_MatchResult],
    ) -> None:
        print(f"目标图: {target_path}")
        print(f"背景图: {background_path}")
        print("\n候选项得分:")
        for result in results:
            print(
                f"  {result.index}. {result.position}: 总分={result.score:.4f}, "
                f"外观={result.appearance_score:.4f}, "
                f"pHash={result.hash_score:.4f}, 像素={result.pixel_score:.4f}, "
                f"结构={result.component_score:.4f}"
            )
