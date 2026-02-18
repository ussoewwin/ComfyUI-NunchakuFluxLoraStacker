/**
 * COLOR Widget for ComfyUI
 *
 * This integration script is licensed under the GNU General Public License v3.0 (GPL-3.0).
 * If you incorporate or modify this code, please credit AILab as the original source:
 * https://github.com/1038lab
 */

import { app } from "/scripts/app.js";

const getContrastTextColor = (hexColor) => {
    if (typeof hexColor !== 'string' || !/^#?[0-9a-fA-F]{6}$/.test(hexColor)) {
        return '#cccccc'; // fallback text color
    }

    const hex = hexColor.replace('#', '');
    const r = parseInt(hex.substr(0, 2), 16);
    const g = parseInt(hex.substr(2, 2), 16);
    const b = parseInt(hex.substr(4, 2), 16);
    const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;

    return luminance > 0.5 ? '#333333' : '#cccccc';
};

const AILabColorWidget = {
    COLORCODE: (key, val) => {
        const widget = {};
        widget.y = 0;
        widget.name = key;
        widget.type = 'COLORCODE';
        widget.options = { default: '#222222' };
        widget.value = typeof val === 'string' ? val : '#222222';

        widget.draw = function (ctx, node, widgetWidth, widgetY, height) {
            const hide = this.type !== 'COLORCODE' && app.canvas.ds.scale > 0.5;
            if (hide) {
                return;
            }

            const margin = 15;
            const verticalMargin = 8;
            const radius = 12;

            ctx.fillStyle = this.value;
            ctx.beginPath();
            const x = margin;
            const y = widgetY + verticalMargin;
            const w = widgetWidth - margin * 2;
            const h = height - verticalMargin * 2;
            ctx.moveTo(x + radius, y);
            ctx.lineTo(x + w - radius, y);
            ctx.quadraticCurveTo(x + w, y, x + w, y + radius);
            ctx.lineTo(x + w, y + h - radius);
            ctx.quadraticCurveTo(x + w, y + h, x + w - radius, y + h);
            ctx.lineTo(x + radius, y + h);
            ctx.quadraticCurveTo(x, y + h, x, y + h - radius);
            ctx.lineTo(x, y + radius);
            ctx.quadraticCurveTo(x, y, x + radius, y);
            ctx.closePath();
            ctx.fill();

            ctx.strokeStyle = '#555';
            ctx.lineWidth = 1;
            ctx.stroke();

            ctx.fillStyle = getContrastTextColor(this.value);
            ctx.font = '12px sans-serif';
            ctx.textAlign = 'center';

            const text = `${this.name} (${this.value})`;
            ctx.fillText(text, widgetWidth * 0.5, widgetY + height * 0.65);
        };

        widget.mouse = function (e, pos, node) {
            if (e.type === 'pointerdown') {
                const margin = 15;

                if (pos[0] >= margin && pos[0] <= node.size[0] - margin) {
                    const picker = document.createElement('input');
                    picker.type = 'color';
                    picker.value = this.value;

                    picker.style.position = 'absolute';
                    picker.style.left = '-9999px';
                    picker.style.top = '-9999px';

                    document.body.appendChild(picker);

                    picker.addEventListener('change', () => {
                        this.value = picker.value;
                        node.graph._version++;
                        node.setDirtyCanvas(true, true);
                        picker.remove();
                    });

                    picker.click();
                    return true;
                }
            }
            return false;
        };

        widget.computeSize = function (width) {
            return [width, 48];
        };

        return widget;
    }
};

app.registerExtension({
    name: "NunchakuSAM3.colorWidgetV2",

    getCustomWidgets(app) {
        return {
            COLORCODE: (node, inputName, inputData) => {
                const defaultValue = inputData?.[1]?.default || inputData?.default || '#222222';
                const colorWidget = AILabColorWidget.COLORCODE(inputName, defaultValue);
                return {
                    widget: colorWidget,
                    minWidth: 150,
                    minHeight: 48,
                };
            }
        };
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // COLORCODE タイプの入力を認識するためにノード定義を拡張
        // ComfyUI nodes 2.0 では nodeData.input の構造が異なる可能性があるため、複数の形式に対応
        const inputs = nodeData.input || {};
        const required = inputs.required || {};
        const optional = inputs.optional || {};
        const allInputs = { ...required, ...optional };
        
        const colorCodeInputs = [];
        for (const [inputName, inputDef] of Object.entries(allInputs)) {
            if (Array.isArray(inputDef) && inputDef[0] === "COLORCODE") {
                colorCodeInputs.push({
                    name: inputName,
                    default: inputDef[1]?.default || '#222222'
                });
            }
        }
        
        if (colorCodeInputs.length > 0) {
            nodeType["@hasColorCode"] = true;
            nodeType["@colorCodeInputs"] = colorCodeInputs;
        }
    },

    nodeCreated(node) {
        // COLORCODE タイプのウィジェットを動的に追加し、一番上に配置
        const nodeType = node.constructor;
        if (nodeType["@hasColorCode"] && nodeType["@colorCodeInputs"]) {
            const colorCodeInputs = nodeType["@colorCodeInputs"];
            const colorWidgets = [];
            
            for (const inputInfo of colorCodeInputs) {
                // 既存のウィジェットを確認
                let widget = node.widgets?.find(w => w.name === inputInfo.name);
                
                if (!widget) {
                    // ウィジェットが存在しない場合は作成
                    const colorWidget = AILabColorWidget.COLORCODE(inputInfo.name, inputInfo.default);
                    try {
                        widget = node.addCustomWidget(colorWidget);
                    } catch (e) {
                        console.warn(`[AILab.colorWidget] Failed to add custom widget for ${inputInfo.name}:`, e);
                        // フォールバック: 通常のウィジェットとして追加を試みる
                        if (node.addWidget) {
                            widget = node.addWidget("text", inputInfo.name, inputInfo.default);
                        }
                    }
                } else {
                    // 既存のウィジェットを COLORCODE タイプに変換
                    if (widget.type !== 'COLORCODE') {
                        const colorWidget = AILabColorWidget.COLORCODE(inputInfo.name, widget.value || inputInfo.default);
                        // 既存のウィジェットを置き換え
                        const index = node.widgets.indexOf(widget);
                        if (index !== -1) {
                            node.widgets[index] = colorWidget;
                            widget = colorWidget;
                        }
                    }
                }
                
                if (widget) {
                    colorWidgets.push(widget);
                }
            }
            
            // カラーパレットウィジェットを一番上に移動
            if (colorWidgets.length > 0 && node.widgets) {
                // 既存のウィジェット配列からカラーパレットウィジェットを削除
                for (const colorWidget of colorWidgets) {
                    const index = node.widgets.indexOf(colorWidget);
                    if (index !== -1) {
                        node.widgets.splice(index, 1);
                    }
                }
                
                // カラーパレットウィジェットを一番上に追加
                node.widgets.unshift(...colorWidgets);
                
                // キャンバスを更新
                if (app.canvas) {
                    app.canvas.setDirty(true, true);
                }
            }
        }
    }
});