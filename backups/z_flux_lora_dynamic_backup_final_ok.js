import { app } from "../../scripts/app.js";

console.log("★★★ z_flux_lora_dynamic.js: PHYSICAL WIDGET RECONSTRUCTION ★★★");

app.registerExtension({
    name: "nunchaku.flux_lora_dynamic_final_fix",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "FluxLoraMultiLoader_10") {
            nodeType["@visibleLoraCount"] = { type: "number", default: 1, min: 1, max: 10, step: 1 };
        }
    },

    nodeCreated(node) {
        if (node.comfyClass !== "FluxLoraMultiLoader_10") return;

        node.serialize_widgets = false;
        if (!node.properties) node.properties = {};
        if (node.properties["visibleLoraCount"] === undefined) node.properties["visibleLoraCount"] = 1;

        // Initialize widget cache
        node.cachedWidgets = {};
        let cacheReady = false;

        // Evacuate all widgets to cache
        const initCache = () => {
            if (cacheReady) return;
            
            // Python definition order: lora_name_1, lora_wt_1, ...
            const all = [...node.widgets];
            
            for (let i = 1; i <= 10; i++) {
                const wName = all.find(w => w.name === `lora_name_${i}`);
                const wWt = all.find(w => w.name === `lora_wt_${i}`);
                if (wName && wWt) {
                    node.cachedWidgets[i] = [wName, wWt];
                    // Enforce types
                    wName.type = "combo";
                    wWt.type = "number";
                    // Remove custom computeSize (revert to default)
                    if (wName.computeSize) delete wName.computeSize;
                    if (wWt.computeSize) delete wWt.computeSize;
                }
            }
            cacheReady = true;
        };

        // Create or get button
        const ensureButton = () => {
            const btnName = "🔢 Set LoRA Count";
            let btn = node.widgets.find(w => w.name === btnName);
            if (!btn) {
                btn = node.addWidget("button", btnName, null, () => {});
            }
            // Set callback
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
            return btn;
        };

        node.updateLoraSlots = function() {
            if (!cacheReady) initCache();

            const count = parseInt(this.properties["visibleLoraCount"] || 1);
            const btn = ensureButton();

            // 1. Physically rebuild widget array (button only first)
            this.widgets = [btn];

            // Add from cache for required count only
            for (let i = 1; i <= count; i++) {
                const pair = this.cachedWidgets[i];
                if (pair) {
                    this.widgets.push(pair[0]); // name
                    this.widgets.push(pair[1]); // wt
                }
            }

            // 2. Height calculation (no extra margin)
            // Only count physically present widgets
            const HEADER_H = 60;
            const SLOT_H = 54; // name(26) + weight(26) + margin
            const PADDING = 20;
            
            // Set height strictly for current count
            const targetH = HEADER_H + (count * SLOT_H) + PADDING;
            
            this.setSize([this.size[0], targetH]);
            
            // Refresh draw
            if (app.canvas) app.canvas.setDirty(true, true);
        };

        node.onPropertyChanged = function(property, value) {
            if (property === "visibleLoraCount") {
                this.updateLoraSlots();
            }
        };

        // Init kick
        setTimeout(() => {
            initCache();
            node.updateLoraSlots();
        }, 100);
    }
});
