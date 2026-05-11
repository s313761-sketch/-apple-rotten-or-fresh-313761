# YOLOv8 GitHub Pages 物件辨識與價值統計網頁

本專案可放在 GitHub Pages 上使用，功能包含：

1. 上傳圖片進行 YOLOv8 物件辨識
2. 啟動手機後鏡頭進行即時辨識
3. 在圖片或影像上框出物件
4. 顯示類別名稱與信心值
5. 統計三種類別數量
6. 統計總數
7. 根據單價計算各類別小計與總價值

---

## 一、檔案結構

```text
index.html
style.css
app.js
labels.json
models/
  best.onnx
README.md
```

---

## 二、重要限制

GitHub Pages 是靜態網站，不能直接執行 Python，也不能直接載入 `best.pt`。

因此你必須先把 YOLOv8 的 `best.pt` 轉成：

```text
best.onnx
```

然後放到：

```text
models/best.onnx
```

---

## 三、Colab 匯出 ONNX

請在 Colab 執行：

```python
!pip install -U ultralytics onnx onnxruntime

from ultralytics import YOLO

model = YOLO("/content/best.pt")
model.export(format="onnx", imgsz=640, opset=12, simplify=True)
```

匯出後會得到：

```text
best.onnx
```

請把它放到本專案的：

```text
models/best.onnx
```

---

## 四、修改類別名稱

請修改 `labels.json`。

目前預設：

```json
[
  "類別1",
  "類別2",
  "類別3"
]
```

若你的三種類別是：

```text
百合甲蟲
荔枝椿象
介殼蟲
```

則改成：

```json
[
  "百合甲蟲",
  "荔枝椿象",
  "介殼蟲"
]
```

注意：順序必須和 Roboflow / YOLOv8 `data.yaml` 裡面的類別順序一致。

---

## 五、修改類別價值

請打開 `app.js`，找到：

```javascript
const CLASS_VALUES = {
  0: 100,
  1: 200,
  2: 300
};
```

代表：

```text
第 0 類：100 元
第 1 類：200 元
第 2 類：300 元
```

請依照你的需求修改。

---

## 六、修改模型輸入尺寸

如果你的模型是用 `imgsz=640` 匯出，保持：

```javascript
const INPUT_SIZE = 640;
```

如果你用 `imgsz=960` 匯出，請同時修改：

```javascript
const INPUT_SIZE = 960;
```

---

## 七、GitHub Pages 使用方式

1. 建立一個 GitHub repository
2. 上傳本專案所有檔案
3. 確認 `models/best.onnx` 已上傳
4. 進入 repository 的 `Settings`
5. 點選 `Pages`
6. Source 選擇 `Deploy from a branch`
7. Branch 選擇 `main`
8. Folder 選擇 `/root`
9. 儲存後等待 GitHub Pages 部署完成

---

## 八、手機鏡頭注意事項

手機鏡頭需要 HTTPS 才能正常啟動。

GitHub Pages 網址是 HTTPS，因此可以使用手機後鏡頭。

若你在本機直接用 `file://` 開啟，鏡頭通常無法使用。

建議本機測試時使用：

```bash
python -m http.server 8000
```

然後用瀏覽器開啟：

```text
http://localhost:8000
```

---

## 九、物件很多時辨識數量偏少

可以先調整網頁上的：

```text
信心值門檻
NMS IoU 門檻
```

建議：

```text
信心值門檻：0.15 ~ 0.25
NMS IoU 門檻：0.70 ~ 0.85
```

若仍然把一群物件視為一個，通常表示訓練資料標註時有把多個物件框成一個，或小物件解析度不足。

建議重新標註與訓練：

1. 每個物件都要單獨框選
2. 框線盡量貼近物件
3. 增加密集物件照片
4. 使用 `imgsz=960` 或 `imgsz=1280` 重新訓練
5. 網頁中的 `INPUT_SIZE` 也要同步修改

---

## 十、常見錯誤

### 1. 顯示模型載入失敗

請確認：

```text
models/best.onnx
```

是否真的存在。

### 2. 類別名稱錯誤

請確認 `labels.json` 順序是否和 YOLOv8 的 `data.yaml` 一致。

### 3. 框的位置不準

請確認：

```javascript
const INPUT_SIZE = 640;
```

是否和 ONNX 匯出時的 `imgsz` 一致。

### 4. 鏡頭不能開

請用 GitHub Pages 的 HTTPS 網址開啟，不要用 `file://`。
