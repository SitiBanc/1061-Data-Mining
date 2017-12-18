# HW5 文章分群與分類
> **本次作業將140篇新聞(共5種主題)斷詞切字後去計算tf-idf值，並以該文章的tf-idf值當特徵值再進行分群或分類**
> **資料集說明: postContent:新聞內文 mainTag:主題類別**

#### 以下為作業流程
### 1. 將資料讀取進來並以[jieba](https://github.com/fxsjy/jieba)(python,要額外下載)套件斷詞,也可以用ckip中文斷詞系統
> 因為中文不像英文每個詞已經分開好了，所以需要斷詞才能計算 tf-idf 值

> 例如：川普當選美國總統。 -> 川普 當選 美國 總統 。

> 文章斷完詞之後會有很多沒有意義的文字(stopword停用詞)，將其刪除之後會計算 tf-idf 值會比較有意義，可以自行建立stopword表或用jieba的關鍵詞抽取

### 2. 用scikit-learn中的[sklearn.feature_extraction.text](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction.text)去計算 tf-idf
> 先將文章的轉成向量在去計算 tf-idf (內建的function)

> 計算完後資料會變成很大的矩陣，每一列代表一篇文章，會有幾百欄，每個欄位都是一個詞，每個欄位的值代表那篇文章中那個詞的 tf-idf 值

> 每篇文章都會有幾百個特徵值

### 3. 進行分群與分類
1. 分群:可使用k-means，設定k=5，最後將每一群print出來，比較其相似程度並做討論
2. 分類:可使用[決策樹](http://napitupulu-jon.appspot.com/posts/decision-tree-ud.html)，將mainTag當作label，訓練75%測試25%，進行分類，最後計算Accuracy


**請上傳**
1. 程式碼
2. WORD或PDF報告
