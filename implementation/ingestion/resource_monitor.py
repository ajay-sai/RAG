"""
Resource monitoring and adaptive ingestion engine selection.
"""

import os
import logging
import psutil
from typing import Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class IngestionMode(Enum):
    """Available ingestion modes based on resource constraints."""
    FULL = "full"  # All features: Whisper Turbo, OCR, contextual enrichment
    STANDARD = "standard"  # Whisper Base, OCR, no contextual enrichment
    LIGHT = "light"  # Whisper Tiny, no OCR, no contextual enrichment
    MINIMAL = "minimal"  # Skip audio/images, text only, no enrichment


class ResourceMonitor:
    """Monitor system resources and recommend ingestion modes."""
    
    # Memory thresholds (GB)
    FULL_MODE_MEMORY_GB = 8.0  # Need 8GB+ for Whisper Turbo
    STANDARD_MODE_MEMORY_GB = 4.0  # Need 4GB+ for Whisper Base
    LIGHT_MODE_MEMORY_GB = 2.0  # Need 2GB+ for Whisper Tiny
    
    # Disk thresholds (GB)
    MIN_DISK_SPACE_GB = 2.0  # Need at least 2GB free
    
    @staticmethod
    def get_system_resources() -> Dict[str, Any]:
        """Get current system resource availability."""
        try:
            # Memory
            memory = psutil.virtual_memory()
            memory_total_gb = memory.total / (1024 ** 3)
            memory_available_gb = memory.available / (1024 ** 3)
            memory_percent = memory.percent
            
            # Disk
            disk = psutil.disk_usage('/')
            disk_total_gb = disk.total / (1024 ** 3)
            disk_free_gb = disk.free / (1024 ** 3)
            disk_percent = disk.percent
            
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            return {
                "memory_total_gb": round(memory_total_gb, 2),
                "memory_available_gb": round(memory_available_gb, 2),
                "memory_percent": memory_percent,
                "disk_total_gb": round(disk_total_gb, 2),
                "disk_free_gb": round(disk_free_gb, 2),
                "disk_percent": disk_percent,
                "cpu_percent": cpu_percent,
                "cpu_count": cpu_count,
            }
        except Exception as e:
            logger.error(f"Failed to get system resources: {e}")
            return {}
    
    @classmethod
    def recommend_ingestion_mode(cls, force_mode: Optional[IngestionMode] = None) -> IngestionMode:
        """
        Recommend ingestion mode based on available resources.
        
        Args:
            force_mode: Override automatic detection with specific mode
            
        Returns:
            Recommended ingestion mode
        """
        if force_mode:
            logger.info(f"Using forced ingestion mode: {force_mode.value}")
            return force_mode
        
        resources = cls.get_system_resources()
        
        if not resources:
            logger.warning("Could not detect resources, defaulting to MINIMAL mode")
            return IngestionMode.MINIMAL
        
        available_memory_gb = resources.get("memory_available_gb", 0)
        free_disk_gb = resources.get("disk_free_gb", 0)
        
        logger.info(f"Available memory: {available_memory_gb:.2f} GB")
        logger.info(f"Free disk space: {free_disk_gb:.2f} GB")
        
        # Check disk space first
        if free_disk_gb < cls.MIN_DISK_SPACE_GB:
            logger.warning(
                f"Low disk space ({free_disk_gb:.2f} GB free), "
                "using MINIMAL mode"
            )
            return IngestionMode.MINIMAL
        
        # Recommend mode based on available memory
        if available_memory_gb >= cls.FULL_MODE_MEMORY_GB:
            logger.info(
                f"Sufficient memory ({available_memory_gb:.2f} GB), "
                "using FULL mode with Whisper Turbo"
            )
            return IngestionMode.FULL
        
        elif available_memory_gb >= cls.STANDARD_MODE_MEMORY_GB:
            logger.info(
                f"Moderate memory ({available_memory_gb:.2f} GB), "
                "using STANDARD mode with Whisper Base"
            )
            return IngestionMode.STANDARD
        
        elif available_memory_gb >= cls.LIGHT_MODE_MEMORY_GB:
            logger.info(
                f"Limited memory ({available_memory_gb:.2f} GB), "
                "using LIGHT mode with Whisper Tiny"
            )
            return IngestionMode.LIGHT
        
        else:
            logger.warning(
                f"Very limited memory ({available_memory_gb:.2f} GB), "
                "using MINIMAL mode (skip audio)"
            )
            return IngestionMode.MINIMAL
    
    @staticmethod
    def get_whisper_model_for_mode(mode: IngestionMode) -> Optional[str]:
        """Get appropriate Whisper model for ingestion mode."""
        whisper_models = {
            IngestionMode.FULL: "WHISPER_TURBO",  # ~3GB, best quality
            IngestionMode.STANDARD: "WHISPER_BASE",  # ~300MB, good quality
            IngestionMode.LIGHT: "WHISPER_TINY",  # ~75MB, basic quality
            IngestionMode.MINIMAL: None,  # Skip audio
        }
        return whisper_models.get(mode)
    
    @staticmethod
    def should_enable_contextual_enrichment(mode: IngestionMode) -> bool:
        """Check if contextual enrichment should be enabled for mode."""
        return mode == IngestionMode.FULL
    
    @staticmethod
    def should_enable_ocr(mode: IngestionMode) -> bool:
        """Check if OCR should be enabled for mode."""
        return mode in [IngestionMode.FULL, IngestionMode.STANDARD]
    
    @staticmethod
    def should_process_audio(mode: IngestionMode) -> bool:
        """Check if audio files should be processed for mode."""
        return mode != IngestionMode.MINIMAL
    
    @classmethod
    def print_resource_summary(cls):
        """Print a summary of system resources and recommendations."""
        resources = cls.get_system_resources()
        mode = cls.recommend_ingestion_mode()
        
        print("\n" + "="*60)
        print("🔍 SYSTEM RESOURCES & INGESTION CONFIGURATION")
        print("="*60)
        
        print("\n📊 Current Resources:")
        print(f"  Memory: {resources.get('memory_available_gb', 0):.2f} GB available "
              f"/ {resources.get('memory_total_gb', 0):.2f} GB total "
              f"({resources.get('memory_percent', 0):.1f}% used)")
        print(f"  Disk:   {resources.get('disk_free_gb', 0):.2f} GB free "
              f"/ {resources.get('disk_total_gb', 0):.2f} GB total "
              f"({resources.get('disk_percent', 0):.1f}% used)")
        print(f"  CPU:    {resources.get('cpu_percent', 0):.1f}% "
              f"({resources.get('cpu_count', 0)} cores)")
        
        print(f"\n🎯 Recommended Mode: {mode.value.upper()}")
        
        whisper_model = cls.get_whisper_model_for_mode(mode)
        print(f"\n⚙️  Configuration:")
        print(f"  Audio Processing:       {'✅ YES' if cls.should_process_audio(mode) else '❌ NO'}")
        if whisper_model:
            print(f"  Whisper Model:          {whisper_model}")
        print(f"  OCR (Images in PDFs):   {'✅ YES' if cls.should_enable_ocr(mode) else '❌ NO'}")
        print(f"  Contextual Enrichment:  {'✅ YES' if cls.should_enable_contextual_enrichment(mode) else '❌ NO'}")
        
        print("\n💡 Mode Descriptions:")
        print("  FULL:     All features (Whisper Turbo, OCR, enrichment) - needs 8GB+ RAM")
        print("  STANDARD: Whisper Base, OCR, no enrichment - needs 4GB+ RAM")
        print("  LIGHT:    Whisper Tiny, no OCR/enrichment - needs 2GB+ RAM")
        print("  MINIMAL:  Skip audio/images, text only - works with any RAM")
        
        print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    # Test resource monitoring
    ResourceMonitor.print_resource_summary()
