import { app } from "../../scripts/app.js";

console.log("★★★ z_flux_lora_dynamic.js: RESTORE COMBO BOX + PHYSICAL DELETE (FINAL) ★★★");

app.registerExtension({
    name: "nunchaku.flux_lora_dynamic_combo_final_restore",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "FluxLoraMultiLoader_10") {
            nodeType["@visibleLoraCount"] = { type: "number", default: 1, min: 1, max: 10, step: 1 };
        }
    },

    nodeCreated(node) {
        if (node.comfyClass !== "FluxLoraMultiLoader_10") return;

        // node.serialize_widgets = false;
        if (!node.properties) node.properties = {};
        if (node.properties["visibleLoraCount"] === undefined) node.properties["visibleLoraCount"] = 1;

        node.cachedWidgets = {};
        let cacheReady = false;

        const initCache = () => {
            if (cacheReady) return;
            const all = [...node.widgets];
            for (let i = 1; i <= 10; i++) {
                const wName = all.find(w => w.name === `lora_name_${i}`);
                const wWt = all.find(w => w.name === `lora_wt_${i}`);
                if (wName && wWt) {
                    node.cachedWidgets[i] = [wName, wWt];
                    wName.type = "combo";
                    wWt.type = "number";
                    if (wName.computeSize) delete wName.computeSize;
                    if (wWt.computeSize) delete wWt.computeSize;
                }
            }
            cacheReady = true;
        };

        const ensureControlWidget = () => {
            const name = "🔢 LoRA Count";
            
            // Remove all buttons
            for (let i = node.widgets.length - 1; i >= 0; i--) {
                const w = node.widgets[i];
                if (w.name === "🔢 Set LoRA Count" || w.type === "button") {
                    node.widgets.splice(i, 1);
                }
            }

            let w = node.widgets.find(x => x.name === name);
            if (!w) {
                const values = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"];
                w = node.addWidget("combo", name, "1", (v) => {
                    const num = parseInt(v);
                    if (!isNaN(num)) {
                        node.properties["visibleLoraCount"] = num;
                        node.updateLoraSlots();
                    }
                }, { values });
            }
            w.value = node.properties["visibleLoraCount"].toString();
            return w;
        };

        node.updateLoraSlots = function() {
            if (!cacheReady) initCache();

            const count = parseInt(this.properties["visibleLoraCount"] || 1);
            const controlWidget = ensureControlWidget();

            // Physical delete (clean layout logic)
            this.widgets = [controlWidget];

            for (let i = 1; i <= count; i++) {
                const pair = this.cachedWidgets[i];
                if (pair) {
                    this.widgets.push(pair[0]); 
                    this.widgets.push(pair[1]);
                }
            }

            // Height calculation
            const HEADER_H = 60;
            const SLOT_H = 54;
            const PADDING = 20;
            const targetH = HEADER_H + (count * SLOT_H) + PADDING;
            
            this.setSize([this.size[0], targetH]);
            
            if (app.canvas) app.canvas.setDirty(true, true);
        };

        node.onPropertyChanged = function(property, value) {
            if (property === "visibleLoraCount") {
                const w = this.widgets.find(x => x.name === "🔢 LoRA Count");
                if (w) w.value = value.toString();
                this.updateLoraSlots();
            }
        };
        
        // Restore on load
        const origOnConfigure = node.onConfigure;
        node.onConfigure = function() {
             if (origOnConfigure) origOnConfigure.apply(this, arguments);
             setTimeout(() => node.updateLoraSlots(), 100);
        };

        setTimeout(() => {
            initCache();
            node.updateLoraSlots();
        }, 100);
    }
});
