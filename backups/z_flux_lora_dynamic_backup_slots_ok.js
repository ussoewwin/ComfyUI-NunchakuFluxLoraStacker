import { app } from "../../scripts/app.js";

console.log("★★★ z_flux_lora_dynamic.js: FORCE TYPE RESTORE & MANUAL HEIGHT ★★★");

const HIDDEN_TAG = "tschide";

// Hardcode correct types to restore (do not rely on origProps)
const WIDGET_TYPES = {
    "lora_name": "combo",
    "lora_wt": "number"
};

app.registerExtension({
    name: "nunchaku.flux_lora_dynamic_restore",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "FluxLoraMultiLoader_10") {
            nodeType["@visibleLoraCount"] = { type: "number", default: 1, min: 1, max: 10, step: 1 };
        }
    },

    nodeCreated(node) {
        if (node.comfyClass !== "FluxLoraMultiLoader_10") return;

        // Avoid serialization issues
        node.serialize_widgets = false;

        if (!node.properties) node.properties = {};
        if (node.properties["visibleLoraCount"] === undefined) node.properties["visibleLoraCount"] = 1;

        node.updateLoraSlots = function() {
            const count = parseInt(this.properties["visibleLoraCount"] || 1);
            
            // 1. Force widget visibility: slots 1..10, visible if i<=count else HIDDEN
            for (let i = 1; i <= 10; i++) {
                const isVisible = i <= count;
                
                // name (combo) and weight (number)
                ["lora_name", "lora_wt"].forEach(prefix => {
                    const wName = `${prefix}_${i}`;
                    const w = this.widgets.find(x => x.name === wName);
                    if (w) {
                        if (isVisible) {
                            // Important: overwrite with correct type, ignore origProps (avoids HIDDEN restore bug)
                            w.type = WIDGET_TYPES[prefix];
                            
                            // Restore default computeSize
                            if (w.computeSize && w.computeSize.toString().includes("return [0, -4]")) {
                                delete w.computeSize; 
                            }
                        } else {
                            w.type = HIDDEN_TAG;
                            // Collapse height
                            w.computeSize = () => [0, -4];
                        }
                    }
                });
            }

            // 2. Manual node height: header + button + (slot count * height)
            // LiteGraph: header~30, button~30, each widget~26; per slot: name(26)+weight(26)+margin ~54px
            const HEADER_H = 60; // includes button
            const SLOT_H = 54; 
            const PADDING = 20;
            
            const targetH = HEADER_H + (count * SLOT_H) + PADDING;
            
            this.setSize([this.size[0], targetH]);
            
            if (app.canvas) app.canvas.setDirty(true, true);
        };

        // Add button
        const btnName = "🔢 Set LoRA Count";
        // Avoid duplicate
        let btn = node.widgets.find(w => w.name === btnName);
        if (!btn) {
            btn = node.addWidget("button", btnName, null, () => {
                const current = node.properties["visibleLoraCount"];
                const val = prompt("Enter LoRA Count (1-10):", current);
                if (val !== null) {
                    const num = parseInt(val);
                    if (!isNaN(num) && num >= 1 && num <= 10) {
                        node.properties["visibleLoraCount"] = num;
                        node.updateLoraSlots();
                    }
                }
            });
        }
        
        // Move button to front (always)
        const btnIdx = node.widgets.indexOf(btn);
        if (btnIdx > 0) {
            node.widgets.splice(0, 0, node.widgets.splice(btnIdx, 1)[0]);
        }
        
        // Re-set callback (reload fix)
        btn.callback = () => {
            const current = node.properties["visibleLoraCount"];
            const val = prompt("Enter LoRA Count (1-10):", current);
            if (val !== null) {
                const num = parseInt(val);
                if (!isNaN(num) && num >= 1 && num <= 10) {
                    node.properties["visibleLoraCount"] = num;
                    node.updateLoraSlots();
                }
            }
        };

        node.onPropertyChanged = function(property, value) {
            if (property === "visibleLoraCount") {
                this.updateLoraSlots();
            }
        };

        // First run
        setTimeout(() => node.updateLoraSlots(), 100);
    }
});
