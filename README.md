# 網多期末專題：偷取 ML-as-Service Model

助教可以跳過偷取前準備工作，直接跑我們產生的 MNIST 結果

## 偷取前準備
### 事前準備：
1. 申請 BigML 帳戶
2. 設定 API 的環境變數 (https://bigml.com/api)
### 準備資料：
1. 準備資料：
將資料存至於 'BigML/data/datasets/' 資料夾下。以 MNIST 為例，此資料夾下已放置 'MNIST/MNIST_train.csv' 以及  'MNIST/MNIST_test.csv'

2. 利用 BigML API 建立 Model：
執行 `python BigML/Create_model.py`
此步驟會完成以下事項
    1. 輸入 Model 名字： Enter a model name:
        * 以 MNIST 為例，輸入 'MNIST'
    2. 輸入資料是否在 Local 端 Local or online (L/O)?
        * 以 MNIST 為例，輸入 'L'
    3. 輸入想用來建立的 model 類型 Which model (Decison_tree / Ensemble / Deepnet / Association_model)?
        * 以 MNIST 為例，輸入 DT/EN/DN/AS
    4. 程式開始執行，資料會上傳到自己帳戶，並且建立 Model
    5. 中途所有call的 api 皆會以 json 檔案格式存取

3. 新增 Model
    1. 進入自己帳號，找到剛剛建立的 model，複製 Model 網址後面的金鑰 (ex. 5c235df4529963147b016a71)
    2. 將新建的 Model 金鑰寫進 'BigML/models' dictionary 裡

4. 獲取 Model 預測資料（之後用來偷取 Model）：
    1. 有分三個檔案： 'BigML/Predict_API_batch_model.py', 'BigML/Predict_API_model.py', 'BigML/Predict_local_model.py'。三個檔案的功能分別為：
        1. 使用線上的 Model，以 batch 的方式做預測
        2. 使用線上的 Model，以一次一個 predict 的方式做預測
        3. 使用local的 Model，以一次一個的方式做預測
    2. 以 MNIST 為例子，執行 `python BigML/Predict_API_batch_model.py`
    3. 輸入 Model 名字。 Enter a model name:
    4. 輸入 Model 的類型。 Which model are u going to predict (Decision tree / Deepnet)?
    5. 完成預測

5. 對產生的預測結果做 Parsing
    1. 執行 `python BigML/data/predict_result/parser.py`
    2. 輸入 Model 的名字 ：Please input model name!
    3. 輸入原本 input 資料的維度：Ur input variable number:
    4. 輸入輸出資料的種類數：Ur output variable category number: 
    5. 結果將存至 'BigML/data/predict_result/'
