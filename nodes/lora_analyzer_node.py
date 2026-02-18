#!/usr/bin/env python3
"""
ComfyUI LoRA Analyzer Node - DIRECT LOAD VERSION
Directly loads LoRA within the node and analyzes it
"""

import os
import json
import glob
import traceback
import hashlib
import urllib.request
from safetensors import safe_open

# ComfyUI imports for LoRA loading
try:
    import comfy.utils
    import comfy.model_management
    import comfy.lora
    import folder_paths
    COMFY_AVAILABLE = True
except ImportError:
    COMFY_AVAILABLE = False
    print("ComfyUI modules not available - running in standalone mode")

class UniversalLoRAAnalyzer:
    """
    ComfyUI LoRA Analyzer Node - Universal Version
    Loads LoRA directly within the node and provides analysis
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get available LoRA files from ComfyUI
        lora_files = ["None"]
        if COMFY_AVAILABLE:
            try:
                lora_files.extend(folder_paths.get_filename_list("loras"))
            except:
                pass
        
        return {
            "required": {
                "lora_name": (lora_files, {"tooltip": "Select LoRA file from dropdown"}),
            },
            "optional": {
                "manual_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Optional: Manual LoRA file path (.safetensors)"
                }),
                "show_training_info": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Show",
                    "label_off": "Hide"
                }),
                "auto_discover": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Enable",
                    "label_off": "Disable"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("model_type", "sub_type", "trigger_words", "base_model", "analysis_result", "lora_info", "civitai_url", "hf_url")
    FUNCTION = "analyze_lora"
    CATEGORY = "LoRA Tools"
    
    def __init__(self):
        pass

    # ... (analyze_lora and other methods remain the same, just indented under the new class name)
    
    def analyze_lora(self, lora_name, manual_path="", show_training_info=True, auto_discover=False):
        """
        Analyze LoRA file directly
        """
        try:
            # Determine which LoRA file to use
            lora_path = self._determine_lora_path(lora_name, manual_path, auto_discover)
            
            if not lora_path:
                return (
                    "No LoRA",
                    "-",
                    "No LoRA file selected or found", 
                    "Unknown", 
                    "Please select a LoRA file from the dropdown or provide a manual path.",
                    "No LoRA loaded",
                    "", ""
                )
            
            # Check if file exists
            if not os.path.exists(lora_path):
                return (
                    "File Not Found",
                    "-",
                    f"File does not exist: {lora_path}", 
                    "Unknown", 
                    f"The specified LoRA file does not exist: {lora_path}",
                    f"File not found: {lora_path}",
                    "", ""
                )
            
            # Analyze the LoRA file
            analysis_result = self._analyze_lora_file(lora_path, show_training_info)
            
            if analysis_result is None:
                return (
                    "Analysis Error",
                    "-",
                    "Failed to analyze LoRA file", 
                    "Unknown", 
                    f"Failed to analyze the LoRA file: {lora_path}",
                    f"Analysis failed: {lora_path}",
                    "", ""
                )
            
            model_type = analysis_result.get('model_type', 'Unknown')
            sub_type = analysis_result.get('sub_type', '-')
            trigger_words = analysis_result.get('trigger_words', 'Unknown')
            base_model = analysis_result.get('base_model', 'Unknown')
            detailed_result = analysis_result.get('detailed_result', 'No analysis result')
            lora_info = f"LoRA analyzed successfully: {os.path.basename(lora_path)}"
            civitai_url = analysis_result.get('civitai_url', '')
            hf_url = analysis_result.get('hf_url', '')
            
            return (model_type, sub_type, trigger_words, base_model, detailed_result, lora_info, civitai_url, hf_url)
            
        except Exception as e:
            error_msg = f"Analysis error: {str(e)}"
            traceback.print_exc()
            return ("Error", "-", error_msg, "Unknown", error_msg, error_msg, "", "")

    def _determine_lora_path(self, lora_name, manual_path, auto_discover):
        """
        Determine which LoRA file to use
        """
        # Priority 1: Manual path
        if manual_path and manual_path.strip():
            manual_path = manual_path.strip()
            if os.path.exists(manual_path) and manual_path.lower().endswith('.safetensors'):
                return manual_path
        
        # Priority 2: Selected from dropdown
        if lora_name and lora_name != "None" and COMFY_AVAILABLE:
            try:
                lora_path = folder_paths.get_full_path("loras", lora_name)
                if lora_path and os.path.exists(lora_path):
                    return lora_path
            except:
                pass
        
        # Priority 3: Auto-discover first available LoRA
        if auto_discover:
            discovered_files = self._discover_lora_files()
            if discovered_files:
                # Extract first file path
                first_line = discovered_files.split('\n')[0]
                if ': ' in first_line:
                    file_path = first_line.split(': ', 1)[1]
                    if os.path.exists(file_path):
                        return file_path
        
        return None
    
    def _discover_lora_files(self):
        """
        Auto-discover LoRA files in common locations
        """
        try:
            discovered = []
            
            # ComfyUI LoRA directory
            if COMFY_AVAILABLE:
                try:
                    lora_dir = folder_paths.get_folder_paths("loras")[0]
                    if os.path.exists(lora_dir):
                        pattern = os.path.join(lora_dir, "*.safetensors")
                        files = glob.glob(pattern)
                        for file_path in files:
                            file_name = os.path.basename(file_path)
                            size_mb = os.path.getsize(file_path) / (1024 * 1024)
                            discovered.append(f"{file_name} ({size_mb:.1f}MB): {file_path}")
                except:
                    pass
            
            # Common LoRA directories
            common_dirs = [
                "models/loras",
                "models/Lora", 
                "../models/loras",
                "../models/Lora",
                "../../models/loras",
                "../../models/Lora",
                os.path.expanduser("~/ComfyUI/models/loras"),
                os.path.expanduser("~/ComfyUI/models/Lora"),
            ]
            
            for base_dir in common_dirs:
                if os.path.exists(base_dir):
                    pattern = os.path.join(base_dir, "*.safetensors")
                    files = glob.glob(pattern)
                    for file_path in files:
                        file_name = os.path.basename(file_path)
                        try:
                            size_mb = os.path.getsize(file_path) / (1024 * 1024)
                            discovered.append(f"{file_name} ({size_mb:.1f}MB): {file_path}")
                        except:
                            discovered.append(f"{file_name}: {file_path}")
            
            # Remove duplicates and sort
            discovered = list(set(discovered))
            discovered.sort()
            
            return "\n".join(discovered[:10])  # Limit to first 10
                
        except Exception as e:
            return f"Error during auto-discovery: {str(e)}"

    def _analyze_lora_file(self, file_path, show_training_info=True):
        """
        Detailed LoRA file analysis
        """
        try:
            with safe_open(file_path, framework="np", device="cpu") as f:
                metadata = f.metadata() if hasattr(f, 'metadata') else {}
                tensor_keys = list(f.keys())
                
                # Determine model type
                model_type, sub_type = self._determine_model_type(tensor_keys, metadata)
                
                # Extract base model
                base_model = self._extract_base_model(metadata)
                
                # Extract trigger words
                trigger_words = self._extract_trigger_words(metadata)
                
                # Extract URLs
                civitai_url, hf_url = self._get_metadata_url(metadata)
                
                # API Lookup if URL missing
                if not civitai_url:
                    try:
                        file_hash = self._get_file_hash(file_path)
                        found_url = self._get_civitai_url_by_hash(file_hash)
                        if found_url:
                            civitai_url = found_url
                    except:
                        pass # Fail silently on network/hash errors
                
                # Generate detailed result
                detailed_result = self._generate_detailed_result(
                    file_path, model_type, sub_type, base_model, trigger_words, 
                    civitai_url, hf_url, metadata, tensor_keys, show_training_info
                )
                
                return {
                    'model_type': model_type,
                    'sub_type': sub_type,
                    'base_model': base_model,
                    'trigger_words': trigger_words,
                    'civitai_url': civitai_url,
                    'hf_url': hf_url,
                    'detailed_result': detailed_result
                }
                
        except Exception as e:
            print(f"LoRA analysis error: {e}")
            traceback.print_exc()
            return None
    
    def _determine_model_type(self, tensor_keys, metadata):
        """
        Advanced model type detection with SDXL sub-classification
        """
        import re
        
        metadata_str = str(metadata).lower() if metadata else ""
        
        # 1. Broad Category Detection
        main_category = "Unknown"
        
        # Key pattern counting
        key_patterns = {
            'sdxl': 0,
            'sd15': 0,
            'flux': 0,
            'qwen': 0
        }
        
        for key in tensor_keys:
            if "lora_unet_output_blocks" in key or "lora_te2" in key or "text_encoder_2" in key or "conditioner" in key:
                key_patterns['sdxl'] += 1
            if "lora_unet_up_blocks" in key and "lora_te2" not in key and "text_encoder_2" not in key:
                key_patterns['sd15'] += 1 
            if "lora_unet_double_blocks" in key or "lora_unet_single_blocks" in key:
                key_patterns['flux'] += 1
            if "transformer_blocks" in key and "ff_net" in key: 
                 key_patterns['qwen'] += 1

        # Refined Logic
        if key_patterns['flux'] > 0:
            main_category = "Flux1"
        elif key_patterns['sdxl'] > 0:
            main_category = "SDXL"
        elif "lora_unet_down_blocks_3_attentions_2" in str(tensor_keys): 
            main_category = "SD 1.5"
        elif key_patterns['qwen'] > 0:
            main_category = "Qwen-Image"
        elif "wan" in metadata_str:
            main_category = "WAN2.2"
        elif key_patterns['sd15'] > 0:
            main_category = "SD 1.5" # Fallback
            
        # 2. Sub-Category Detection
        sub_category = "-"
        
        if main_category == "SDXL":
            # Check for Pony
            is_pony = False
            
            base_model_name = metadata.get("ss_sd_model_name", "").lower() if metadata else ""
            if "pony" in base_model_name:
                is_pony = True
            if "290640" in base_model_name: # Civitai Hash for Pony V6
                is_pony = True
            if "autismmix" in base_model_name:
                is_pony = True
            
            # Regex check avoids "ponytail" false positives
            if not is_pony and re.search(r'\\bpony\\b', metadata_str):
                 is_pony = True
                 
            if is_pony:
                sub_category = "Pony"
            else:
                # Check for Illustrious / Animagine
                is_illust = False
                if "illustrious" in base_model_name or "animagine" in base_model_name:
                    is_illust = True
                if "illustrious" in metadata_str or "animagine" in metadata_str:
                    is_illust = True
                
                if is_illust:
                    sub_category = "Illustration"
                else:
                    sub_category = "SDXL 1.0"

        return main_category, sub_category
    
    def _extract_base_model(self, metadata):
        """
        Extract base model information
        """
        if not metadata:
            return "Unknown"
        
        keys = ['ss_base_model_version', 'ss_sd_model_name', 'modelspec.sai_model_spec', 'base_model']
        for k in keys:
            if k in metadata and metadata[k] and str(metadata[k]).lower() != "none":
                return str(metadata[k])
        return "Unknown"
    
    
    def _extract_trigger_words(self, metadata):
        """
        Extract trigger words
        """
        if not metadata:
            return "None"
        
        triggers = []
        
        # ss_tag_frequency
        if 'ss_tag_frequency' in metadata:
            try:
                js = json.loads(metadata['ss_tag_frequency'])
                for _, tags in js.items():
                    # Sort by freq
                    sorted_tags = sorted(tags.items(), key=lambda x: x[1], reverse=True)
                    for tag, freq in sorted_tags:
                        if freq > 1: # Threshold
                            triggers.append(tag)
            except:
                pass
                
        # modelspec.trigger_phrase
        if 'modelspec.trigger_phrase' in metadata:
            triggers.append(metadata['modelspec.trigger_phrase'])
            
        # Unique and Clean
        seen = set()
        clean_triggers = []
        for t in triggers:
            t_clean = t.strip()
            if t_clean and t_clean not in seen:
                clean_triggers.append(t_clean)
                seen.add(t_clean)
                
        return ", ".join(clean_triggers[:10]) if clean_triggers else "None"
    
    def _get_metadata_url(self, metadata):
        """
        Extract URLs from metadata
        """
        if not metadata:
            return "", ""
            
        civitai = ""
        hf = ""
        
        # Check common keys
        for k, v in metadata.items():
            v_str = str(v).lower()
            if "civitai.com" in v_str:
                civitai = str(v)
            if "huggingface.co" in v_str:
                hf = str(v)
                
        return civitai, hf

    def _get_file_hash(self, file_path):
        """Calculate SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest().upper()

    def _get_civitai_url_by_hash(self, file_hash):
        """Retrieve Civitai model URL using file hash (using urllib)."""
        try:
            url = f"https://civitai.com/api/v1/model-versions/by-hash/{file_hash}"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode('utf-8'))
                    if 'modelId' in data:
                        return f"https://civitai.com/models/{data['modelId']}?modelVersionId={data['id']}"
        except Exception:
            pass
        return ""

    def _generate_detailed_result(self, file_path, model_type, sub_type, base_model, 
                                trigger_words, civitai_url, hf_url, metadata, tensor_keys, show_training_info):
        """
        Generate detailed analysis result
        """
        result = []
        result.append("=== LoRA Analysis Result (V2) ===")
        result.append(f"File: {os.path.basename(file_path)}")
        result.append(f"Path: {file_path}")
        result.append(f"Model Type: {model_type}")
        result.append(f"Sub Type: {sub_type}")
        result.append(f"Base Model: {base_model}")
        result.append("")
        
        # URLs
        if civitai_url:
            result.append(f"Civitai URL: {civitai_url}")
        if hf_url:
            result.append(f"HuggingFace URL: {hf_url}")
        if civitai_url or hf_url:
            result.append("")

        # Trigger words
        if ", " in trigger_words:
            words = trigger_words.split(", ")
            result.append(f"Trigger Words ({len(words)}):")
            for i, word in enumerate(words, 1):
                result.append(f"   {i}. {word}")
        else:
            result.append(f"Trigger Words: {trigger_words}")
        result.append("")
        
        # Training information
        if show_training_info and metadata:
            result.append("Training Information:")
            training_keys = [
                ('ss_network_alpha', 'Network Alpha'),
                ('ss_network_dim', 'Network Dimension'),
                ('ss_num_epochs', 'Epochs'),
                ('ss_num_train_images', 'Training Images'),
                ('ss_sd_model_name', 'Base Model Name'),
            ]
            
            for key, label in training_keys:
                if key in metadata:
                    value = metadata[key]
                    if value is not None and str(value) != "None":
                        result.append(f"  {label}: {value}")
            result.append("")
        
        # Technical information
        result.append("Technical Information:")
        result.append(f"  Tensor Count: {len(tensor_keys)}")
        result.append(f"  File Size: {os.path.getsize(file_path) / (1024*1024):.1f} MB")
        
        return "\n".join(result)

# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "UniversalLoRAAnalyzer": UniversalLoRAAnalyzer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniversalLoRAAnalyzer": "Universal LoRA Analyzer"
}
