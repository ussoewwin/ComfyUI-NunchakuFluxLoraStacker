import { app } from "../../scripts/app.js";

// Constants - Copied from rgthree
const MODE_ALWAYS = 0;
const MODE_MUTE = 2;
const MODE_BYPASS = 4;

// Property Keys - Copied from rgthree
const PROPERTY_SORT = "sort";
const PROPERTY_SORT_CUSTOM_ALPHA = "customSortAlphabet";
const PROPERTY_MATCH_COLORS = "matchColors";
const PROPERTY_MATCH_TITLE = "matchTitle";
const PROPERTY_SHOW_NAV = "showNav";
const PROPERTY_SHOW_ALL_GRAPHS = "showAllGraphs";
const PROPERTY_RESTRICTION = "toggleRestriction";
const PROPERTY_MODE = "effectMode";

// Simple service to schedule refresh - Simplified implementation of rgthree's FAST_GROUPS_SERVICE
class SimpleRefreshService {
    constructor() {
        this.nodes = [];
        this.scheduled = false;
    }
    
    addNode(node) {
        if (!this.nodes.includes(node)) {
            this.nodes.push(node);
            this.scheduleRefresh();
        }
    }
    
    removeNode(node) {
        const index = this.nodes.indexOf(node);
        if (index > -1) {
            this.nodes.splice(index, 1);
        }
    }
    
    scheduleRefresh() {
        if (this.scheduled) return;
        this.scheduled = true;
        setTimeout(() => {
            this.refresh();
            this.scheduled = false;
            if (this.nodes.length > 0) {
                this.scheduleRefresh();
            }
        }, 100);
    }
    
    refresh() {
        for (const node of this.nodes) {
            if (!node.removed) {
                node.refreshWidgets();
            }
        }
    }
}

const SERVICE = new SimpleRefreshService();

app.registerExtension({
    name: "nunchakufluxlorastacker.fast_groups_bypass_v2.fixed",
    
    // Same as rgthree: Property definition in beforeRegisterNodeDef
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "FastGroupsBypasserV2") {
            // Property definitions - Exact copy from rgthree
                nodeType["@matchColors"] = { type: "string" };
                nodeType["@matchTitle"] = { type: "string" };
                nodeType["@showNav"] = { type: "boolean" };
                nodeType["@showAllGraphs"] = { type: "boolean" };
            nodeType["@sort"] = {
                type: "combo",
                values: ["position", "alphanumeric", "custom alphabet"]
            };
            nodeType["@customSortAlphabet"] = { type: "string" };
            nodeType["@toggleRestriction"] = {
                type: "combo",
                values: ["default", "max one", "always one"]
            };
            nodeType["@effectMode"] = {
                type: "combo",
                values: ["Bypass", "Mute"]
            };

            // Add items to right-click menu (add to prototype)
            const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
            nodeType.prototype.getExtraMenuOptions = function(canvas, options) {
                const r = origGetExtraMenuOptions ? origGetExtraMenuOptions.apply(this, arguments) : options;
                
                r.push(
                    null, // Separator
                    {
                        content: "🎨 Edit Match Colors",
                        callback: () => {
                            const currentValue = this.properties[PROPERTY_MATCH_COLORS] || "";
                            const newValue = prompt("Match Colors (comma separated, e.g. red,blue,#ff0000):", currentValue);
                            if (newValue !== null) {
                                this.properties[PROPERTY_MATCH_COLORS] = newValue;
                                this.onPropertyChanged && this.onPropertyChanged(PROPERTY_MATCH_COLORS, newValue);
                            }
                        }
                    },
                    {
                        content: "📝 Edit Match Title",
                        callback: () => {
                            const currentValue = this.properties[PROPERTY_MATCH_TITLE] || "";
                            const newValue = prompt("Match Title (regex pattern):", currentValue);
                            if (newValue !== null) {
                                this.properties[PROPERTY_MATCH_TITLE] = newValue;
                                this.onPropertyChanged && this.onPropertyChanged(PROPERTY_MATCH_TITLE, newValue);
                            }
                        }
                    }
                );
                
                return r;
            };
        }
    },

    // Same as rgthree: Initialization in nodeCreated
    nodeCreated(node) {
        if (node.comfyClass !== "FastGroupsBypasserV2") return;

        // Property initialization - Exact copy from rgthree
        if (!node.properties) node.properties = {};
        if (node.properties[PROPERTY_MATCH_COLORS] === undefined) node.properties[PROPERTY_MATCH_COLORS] = "";
        if (node.properties[PROPERTY_MATCH_TITLE] === undefined) node.properties[PROPERTY_MATCH_TITLE] = "";
        if (node.properties[PROPERTY_SHOW_NAV] === undefined) node.properties[PROPERTY_SHOW_NAV] = true;
        if (node.properties[PROPERTY_SHOW_ALL_GRAPHS] === undefined) node.properties[PROPERTY_SHOW_ALL_GRAPHS] = true;
        if (node.properties[PROPERTY_SORT] === undefined) node.properties[PROPERTY_SORT] = "position";
        if (node.properties[PROPERTY_SORT_CUSTOM_ALPHA] === undefined) node.properties[PROPERTY_SORT_CUSTOM_ALPHA] = "";
        if (node.properties[PROPERTY_RESTRICTION] === undefined) node.properties[PROPERTY_RESTRICTION] = "default";
        if (node.properties[PROPERTY_MODE] === undefined) node.properties[PROPERTY_MODE] = "Bypass";
        
        node.serialize_widgets = false;
        node.removed = false;
        
        node.refreshWidgets = function() {
            if (!app || !app.graph) return;
            const graph = app.graph;
            
            let groups = [];
            if (graph._groups) groups = [...graph._groups];
            
            const sortMode = this.properties[PROPERTY_SORT] || "position";
            if (sortMode === "custom alphabet") {
                const alphaStr = (this.properties[PROPERTY_SORT_CUSTOM_ALPHA] || "").replace(/\n/g, "");
                if (alphaStr && alphaStr.trim()) {
                    const alphabet = alphaStr.includes(",") 
                        ? alphaStr.toLowerCase().split(",").map(s => s.trim())
                        : alphaStr.toLowerCase().trim().split("");
                    
                groups.sort((a, b) => {
                    const titleA = (a.title || "").toLowerCase();
                    const titleB = (b.title || "").toLowerCase();
                    let idxA = alphabet.findIndex(prefix => titleA.startsWith(prefix));
                    let idxB = alphabet.findIndex(prefix => titleB.startsWith(prefix));
                        if (idxA !== -1 && idxB !== -1) {
                            const ret = idxA - idxB;
                            if (ret === 0) return titleA.localeCompare(titleB);
                            return ret;
                        }
                    if (idxA !== -1) return -1;
                    if (idxB !== -1) return 1;
                    return titleA.localeCompare(titleB);
                });
                }
            } else if (sortMode === "alphanumeric") {
                groups.sort((a, b) => (a.title || "").localeCompare(b.title || ""));
            } else {
                groups.sort((a, b) => {
                    if (Math.abs(a.pos[1] - b.pos[1]) > 50) return a.pos[1] - b.pos[1];
                    return a.pos[0] - b.pos[0];
                });
            }

            let filterColors = (this.properties[PROPERTY_MATCH_COLORS] || "").split(",").filter(c => c.trim());
            if (filterColors.length) {
                filterColors = filterColors.map(color => {
                    color = color.trim().toLowerCase();
                    if (typeof LGraphCanvas !== "undefined" && LGraphCanvas.node_colors && LGraphCanvas.node_colors[color]) {
                        color = LGraphCanvas.node_colors[color].groupcolor;
                    }
                    color = color.replace("#", "").toLowerCase();
                    if (color.length === 3) {
                        color = color.replace(/(.)(.)(.)/, "$1$1$2$2$3$3");
                    }
                    return `#${color}`;
                });
            }
            
            let filteredGroups = [];
            for (const group of groups) {
                if (filterColors.length) {
                    let groupColor = (group.color || "").replace("#", "").trim().toLowerCase();
                    if (!groupColor) continue;
                    if (groupColor.length === 3) {
                        groupColor = groupColor.replace(/(.)(.)(.)/, "$1$1$2$2$3$3");
                    }
                    groupColor = `#${groupColor}`;
                    if (!filterColors.includes(groupColor)) continue;
                }
                
                const matchTitle = (this.properties[PROPERTY_MATCH_TITLE] || "").trim();
                if (matchTitle) {
                    try {
                        if (!new RegExp(matchTitle, "i").test(group.title)) continue;
                    } catch (e) {
                        console.error(e);
                        continue;
                    }
                }
                
                const showAllGraphs = this.properties[PROPERTY_SHOW_ALL_GRAPHS];
                if (!showAllGraphs && graph !== app.canvas.graph) {
                    continue;
                }
                
                filteredGroups.push(group);
            }

            let index = 0;
            
            let editColorsBtn = (this.widgets || []).find(w => w.name === "🎨 Edit Match Colors");
            if (!editColorsBtn) {
                editColorsBtn = this.addWidget("button", "🎨 Edit Match Colors", null, () => {
                    const currentValue = this.properties[PROPERTY_MATCH_COLORS] || "";
                    const newValue = prompt("Match Colors (comma separated, e.g. red,blue,#ff0000):", currentValue);
                    if (newValue !== null) {
                        this.properties[PROPERTY_MATCH_COLORS] = newValue;
                        this.refreshWidgets();
                    }
                });
            }
            if (editColorsBtn && this.widgets && this.widgets[index] !== editColorsBtn) {
                const oldIndex = this.widgets.findIndex(w => w === editColorsBtn);
                if (oldIndex !== -1) {
                    this.widgets.splice(index, 0, this.widgets.splice(oldIndex, 1)[0]);
                }
            }
            index++;
            
            let editTitleBtn = (this.widgets || []).find(w => w.name === "📝 Edit Match Title");
            if (!editTitleBtn) {
                editTitleBtn = this.addWidget("button", "📝 Edit Match Title", null, () => {
                    const currentValue = this.properties[PROPERTY_MATCH_TITLE] || "";
                    const newValue = prompt("Match Title (regex pattern):", currentValue);
                    if (newValue !== null) {
                        this.properties[PROPERTY_MATCH_TITLE] = newValue;
                        this.refreshWidgets();
                    }
                });
            }
            if (editTitleBtn && this.widgets && this.widgets[index] !== editTitleBtn) {
                const oldIndex = this.widgets.findIndex(w => w === editTitleBtn);
                if (oldIndex !== -1) {
                    this.widgets.splice(index, 0, this.widgets.splice(oldIndex, 1)[0]);
                }
            }
            index++;
            
            for (const group of filteredGroups) {
                const title = group.title || "Group";
                const widgetName = `Enable: ${title}`;
                
                let widget = this.widgets ? this.widgets.find(w => w.name === widgetName) : null;
                
                if (!widget) {
                    const isEnabled = isGroupEnabled.call(this, group);
                    widget = this.addWidget("toggle", widgetName, isEnabled, (v) => {});
                }
                
                if (widget) {
                    widget.callback = (v) => handleToggle.call(this, group, v, widget);
                    
                    if (this.widgets[index] !== widget) {
                        const oldIndex = this.widgets.findIndex(w => w === widget);
                        if (oldIndex !== -1) {
                            this.widgets.splice(index, 0, this.widgets.splice(oldIndex, 1)[0]);
                        }
                    }
                }
                
                index++;
            }
            
            while ((this.widgets || [])[index]) {
                this.removeWidget(this.widgets[index]);
            }

            this.setSize(this.computeSize());
            if (app.canvas) app.canvas.setDirty(true, true);
        };

        // Helper functions
        function getNodesInGroup(group) {
            const graph = app.graph;
            if (!graph) return [];
            const nodes = [];
            if (graph._nodes) {
                graph._nodes.forEach(n => {
                    if (!n.pos || !n.size) return;
                    const cx = n.pos[0] + n.size[0] / 2;
                    const cy = n.pos[1] + n.size[1] / 2;
                    if (cx >= group.pos[0] && cx <= group.pos[0] + group.size[0] &&
                        cy >= group.pos[1] && cy <= group.pos[1] + group.size[1]) {
                        nodes.push(n);
                    }
                });
            }
            return nodes;
        }

        function isGroupEnabled(group) {
            const nodes = getNodesInGroup(group);
            if (nodes.length === 0) return true; 
            // Check if ANY node is active (MODE_ALWAYS). 
            // If so, the toggle should be ON.
            return nodes.some(n => n.mode === MODE_ALWAYS);
        }

        function setGroupMode(group, enable) {
            const graph = app.graph;
            if (!graph) return;
            
            const modeSetting = this.properties[PROPERTY_MODE];
            const modeOff = (modeSetting === "Mute") ? MODE_MUTE : MODE_BYPASS;
            const targetMode = enable ? MODE_ALWAYS : modeOff;
            
            const nodes = getNodesInGroup(group);
            let changed = false;
            
            // IMPORTANT: Batch operation to prevent multiple graph updates
            graph.change(); // Begin transaction
            
            nodes.forEach(n => {
                if (n.id !== this.id && n.mode !== targetMode) {
                    n.mode = targetMode;
                    changed = true;
                }
            });
            
            graph.change(); // End transaction
            
            if (changed) {
                // Use requestAnimationFrame to coalesce updates
                requestAnimationFrame(() => {
                   app.canvas.setDirty(true, true);
                });
            }
        }

        function handleToggle(group, newValue, widget) {
            const restriction = this.properties[PROPERTY_RESTRICTION];
            
            if (newValue && (restriction === "max one" || restriction === "always one")) {
                if (this.widgets) {
                    this.widgets.forEach(w => {
                        if (w !== widget && w.type === "toggle" && w.name.startsWith("Enable: ") && w.value) {
                            w.value = false;
                        }
                    });
                }
                applyAllWidgets.call(this);
            } else if (!newValue && restriction === "always one") {
                const anyOn = this.widgets ? this.widgets.some(w => w.type === "toggle" && w.name.startsWith("Enable: ") && w.value) : false;
                if (!anyOn) {
                    widget.value = true;
                    newValue = true;
                }
            }

            setGroupMode.call(this, group, newValue);
            if (restriction !== "default") applyAllWidgets.call(this);
        }

        function applyAllWidgets() {
            const graph = app.graph;
            if (!graph) return;
            const groups = graph._groups || [];
            
            if (this.widgets) {
                this.widgets.forEach(w => {
                    if (w.type === "toggle" && w.name.startsWith("Enable: ")) {
                        const title = w.name.substring(8);
                        const g = (groups || []).find(grp => (grp.title || "Group") === title);
                        if (g) setGroupMode.call(this, g, w.value);
                    }
                });
            }
        }

        // Immediate refresh on property change
        node.onPropertyChanged = function(property, value) {
            if ([PROPERTY_MATCH_COLORS, PROPERTY_MATCH_TITLE, PROPERTY_SORT, PROPERTY_SORT_CUSTOM_ALPHA].includes(property)) {
                this.refreshWidgets();
            }
            if (property === PROPERTY_MODE) {
                const graph = app.graph;
                if (graph && this.widgets) {
                    this.widgets.forEach(w => {
                        if (w.type === "toggle" && w.name.startsWith("Enable: ")) {
                            const title = w.name.substring(8);
                            const g = (graph._groups || []).find(grp => (grp.title || "Group") === title);
                            if (g) setGroupMode.call(this, g, w.value);
                        }
                    });
                }
            }
            return true;
        };

        // Right-click menu items already added in beforeRegisterNodeDef

        // Same as rgthree: Register to service
        const origOnAdded = node.onAdded;
        node.onAdded = function(graph) {
            SERVICE.addNode(this);
            if (origOnAdded) origOnAdded.call(this, graph);
        };

        const origOnRemoved = node.onRemoved;
        node.onRemoved = function() {
            this.removed = true;
            SERVICE.removeNode(this);
            if (origOnRemoved) origOnRemoved.call(this);
        };

        // Add property editing button widgets first (so they are not removed)
        setTimeout(() => {
            node.refreshWidgets();
            if (node.graph) {
                SERVICE.addNode(node);
            }
        }, 100);
    }
});
