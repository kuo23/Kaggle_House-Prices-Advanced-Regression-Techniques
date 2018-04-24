
# [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)


# 一、資料介紹

1. 資料集介紹 

    這個資料集是關於美國愛荷華州Ames的房價資料，以及眾多變數關於房屋的資訊，藉此希望我們能夠建立預測房價的有效模型。

2. 變數介紹 
   
    資料中有79個變數可以做為預測的輸入值。其中包含面積、各樓層面積、房屋用料品質、住宅類型、所處位置等等。    
    
    *變數一覽表*
  
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 81 columns</p>
</div>


# 二、資料探索與變數分析
  
為了能夠挑選出真正有用的資訊，與提高模型的效率，我們對於變數作了以下分析，分別是

- 分析預測目標
- 資料探索
- 特徵分析
- 特徵工程與缺值填補


## 分析預測目標： 

首先我們必須了解有關於要預測的目標：最終房價的資料的分布情形。若是以數據量化： 
- Skewness: 1.882876
- Kurtosis: 6.536282
    
![png](https://i.imgur.com/Y4ELSmC.png)
    
可以看出整體的趨勢為**右偏**，因為迴歸分析盡量要讓預測目標是常態分配，等等我們會將銷售價格進行對數化處理。


## 資料探索

可以先來簡單探索特徵與預測值之間的關係，先提出兩個假設：
- 生活起居面積(**GrLivArea**)越大，房價會越貴

![生活起居vs房價](https://i.imgur.com/p1q1vy0.png)

- 房子用料與整體品質(**OverallQual**)越好，房價會越貴

![房屋整體品質vs房價](https://i.imgur.com/gnC9mLP.png)


## 特徵分析

因為但是變數很多，不可能一個一個去畫散佈圖，必須挑選出較為重要的變數，故我們先依**相關係數矩陣圖**判斷變數間的關係。
         
![png](https://i.imgur.com/EylRj6U.png)

其中相關係數前10高的變數：

![png](https://i.imgur.com/z1AQlKx.png)

舉例來說，可以看到**GarageCars**(車庫放車空間)跟**GarageArea**(車庫面積)的相關性太高，可能會有共線性問題，但是**GarageCars**跟**Y**的相關係數較高，故取**GarageCars**。


以此類推：

- **TotalBsmtSF**(總地下室面積)與**1stFlrSF**(一樓面積)相關係數太高，取**TotalBsmtSF**
- **TotRmsAbvGrd**(房間總數)跟**GrLivArea**(地面上可用空間)相關係數太高，取**GrLivArea**

## 特徵工程與缺值處理

我們以下列原則處理：
- 將變數分為類別型與數值型
- 缺值太多的資料：刪掉
- 缺值不多的資料：直接填入眾數/中位數/平均數

我們將每個特徵缺值佔所有樣本的百分比做成表格，以利進行分析。

*訓練集缺值一覽表*
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Total</th>
      <th>Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>PoolQC</th>
      <td>1453</td>
      <td>0.995205</td>
    </tr>
    <tr>
      <th>MiscFeature</th>
      <td>1406</td>
      <td>0.963014</td>
    </tr>
    <tr>
      <th>Alley</th>
      <td>1369</td>
      <td>0.937671</td>
    </tr>
    <tr>
      <th>Fence</th>
      <td>1179</td>
      <td>0.807534</td>
    </tr>
    <tr>
      <th>FireplaceQu</th>
      <td>690</td>
      <td>0.472603</td>
    </tr>
    <tr>
      <th>LotFrontage</th>
      <td>259</td>
      <td>0.177397</td>
    </tr>
    <tr>
      <th>GarageQual</th>
      <td>81</td>
      <td>0.055479</td>
    </tr>
    <tr>
      <th>GarageCond</th>
      <td>81</td>
      <td>0.055479</td>
    </tr>
    <tr>
      <th>GarageFinish</th>
      <td>81</td>
      <td>0.055479</td>
    </tr>
    <tr>
      <th>GarageType</th>
      <td>81</td>
      <td>0.055479</td>
    </tr>
    <tr>
      <th>GarageYrBlt</th>
      <td>81</td>
      <td>0.055479</td>
    </tr>
    <tr>
      <th>BsmtExposure</th>
      <td>38</td>
      <td>0.026027</td>
    </tr>
    <tr>
      <th>BsmtFinType2</th>
      <td>38</td>
      <td>0.026027</td>
    </tr>
    <tr>
      <th>BsmtQual</th>
      <td>37</td>
      <td>0.025342</td>
    </tr>
    <tr>
      <th>BsmtCond</th>
      <td>37</td>
      <td>0.025342</td>
    </tr>
    <tr>
      <th>BsmtFinType1</th>
      <td>37</td>
      <td>0.025342</td>
    </tr>
    <tr>
      <th>MasVnrArea</th>
      <td>8</td>
      <td>0.005479</td>
    </tr>
    <tr>
      <th>MasVnrType</th>
      <td>8</td>
      <td>0.005479</td>
    </tr>
    <tr>
      <th>Electrical</th>
      <td>1</td>
      <td>0.000685</td>
    </tr>
    <tr>
      <th>RoofStyle</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>

### 訓練集缺值處理

- **PoolQC**(游泳池品質)、**MiscFeature**(其他類別未涉及功能)、**Alley**(通道類型)、**Fence**(柵欄品質)、**FireplaceQu**(壁爐品質)缺值太多，可以直接刪除

- **Garage族群**缺值數量都相同，而且我們覺得這個指標已經可以很好的用「**GarageCars**」這個指標來表達，所以可以考慮都刪除

- 同理，**Bsmt族群(地下室)**都可以考慮皆以**TotalBsmtSF**表達

- **MasVnr族群**是磚石路面的特徵，我們覺得這可以用OverallQual（總體材料與加工質量）代表，且我們認為一般人購屋並不會考慮到這一點，故予以刪除。
- 依照諸如此類的準則，將變數刪減至37個。

### 類別轉換數值

將全資料類別變數轉為數值（1, 2, 3...），並將偏態的變數（建築面積等等）取log，使變數更接近鐘型分配。
    
```python
for i in feature:
    if  type(test[i][0])== str:
        test[i] = test[i].astype('category').cat.codes
```

```
test['logLotArea']=test['LotArea'].apply(math.log)
```

### 測試集缺值填補

- 數值型範例：**BsmtUnfSF**(尚未完成的地下室面積)
```python
test['BsmtUnfSF'].median()
```
```
460.0
```
```python
test['BsmtUnfSF'][test['BsmtUnfSF'].isnull()]=460
```

  
- 類別型範例：**MSzoning**(建物分區分類)

![png](https://i.imgur.com/ROp7qrU.png)
  
```python
test['MSzoning']
test['MSzoning'][test['MSzoning'].isnull()]='RL'
``` 
  
### 最終選取特徵
```
feature=['MSSubClass', 'MSZoning', 'LotArea','LotShape',
       'LotConfig', 'Neighborhood','BldgType', 'HouseStyle', 
       'OverallQual','OverallCond', 'YearBuilt', 'YearRemodAdd', 
       'Foundation','BsmtUnfSF', 'TotalBsmtSF','HeatingQC', 
       'CentralAir', 'Electrical', '2ndFlrSF','LowQualFinSF',
       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'Fireplaces', 'GarageCars','WoodDeckSF', 'OpenPorchSF',
       'PoolArea', 'MiscVal', 'SaleType','SaleCondition']
```

# 三、 模型設定

 ```
 regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(x_train, y_train)
```

```python
regr.score(x_train,y_train)
```
```
0.67200150087312815
```

## 利用交叉驗證及grid_search找出隨機森林最好的參數

```python
param_test = {
    'n_estimators': range(100, 300,100),
    'max_depth': range(2, 7, 1)
}
grid_search = GridSearchCV(estimator = regr, param_grid = param_test, 
scoring='neg_mean_squared_error', cv=5)

grid_search.fit(x_train, y_train)
grid_search.best_params_, grid_search.best_score_
```

得到最好的參數為：樹最大深度為6，200顆弱迴歸樹

    ({'max_depth': 6, 'n_estimators': 200}, -0.023381842324426236)

## 畫出重要特徵 


![png](https://i.imgur.com/9YWFLHx.png)


上傳到Kaggle後，分數為**0.15710**，名次為**2780/5052**。

# 四、檢討

- 人為手動挑選刪除掉太多特徵，可能刪除掉過多有用資訊。
- 訓練集跟測試集要先合併進行資料處理，要訓練時再切開，避免誤差。
- 特徵工程粗糙，直接丟掉缺失值，例如游泳池品質雖然缺失值很多，但是游泳池品質好可能會讓房價高一點，應該讓「None」變成其中一個分類。
- 沒有做dummy，純粹把類別變數變成12345等等...這些有些類別變數地位是相等的，沒有好壞之分，但是轉成數字後，1和5就有距離上的差異，會造成學習誤差。

---

# 重新優化

# 一、 特徵工程改進

## 共同處理

這次我們將測試集與訓練集先併在一起處理缺失值與特徵工程，再分割使用。

## y變數處理

- 刪除離群值

為了提高預測的準確率，我們需要對預測變數y做更完整的處理。    
再看看一次**GrLivAream**(地面上生活區面積)與y的關係。

![img](https://i.imgur.com/skBcEoP.png)

由面積與房價的價格散佈圖可以看出，右下角有些樣本有非常大的面積，可能因為某些原因卻沒有那麼高的價格，我們決定將這些樣本視為例外排除。
(我們定義為**面積大於4000且不到300000者**為離群值)。

排除之後會讓單調性趨勢更加明顯。
![img](https://i.imgur.com/a14XlQz.png)


- 常態分配化
 
原資料
![img](https://i.imgur.com/LJcuJ45.png)
修正後
![img](https://i.imgur.com/nTQtv2A.png)

原資料QQ-plot   
![img](https://i.imgur.com/C15GljW.png)
修正後QQ-plot    
![img](https://i.imgur.com/9mPSMXN.png)


### x變數修改

- 填入缺值

先來看看每個變數的缺值分布情形。

![img](https://i.imgur.com/270JX4r.png) 

我們決定不依缺值量刪除變數，而是將空值作為特徵的一環放入模型(填入None)。

- 標準化 
類別型的X變數，依Scikit-Learn的模型LabelEncoder進行標準化。 
數值型的X變數，則先計算其偏態係數       
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Skew</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MiscVal</th>
      <td>21.940</td>
    </tr>
    <tr>
      <th>PoolArea</th>
      <td>17.689</td>
    </tr>
    <tr>
      <th>LotArea</th>
      <td>13.109</td>
    </tr>
    <tr>
      <th>LowQualFinSF</th>
      <td>12.085</td>
    </tr>
    <tr>
      <th>3SsnPorch</th>
      <td>11.372</td>
    </tr>
    <tr>
      <th>LandSlope</th>
      <td>4.973</td>
    </tr>
    <tr>
      <th>KitchenAbvGr</th>
      <td>4.301</td>
    </tr>
    <tr>
      <th>BsmtFinSF2</th>
      <td>4.145</td>
    </tr>
    <tr>
      <th>EnclosedPorch</th>
      <td>4.002</td>
    </tr>
    <tr>
      <th>ScreenPorch</th>
      <td>3.945</td>
    </tr>
  </tbody>
</table>
</div>

其中絕對值大於0.75的變數(59筆)則取 Box-Cox 轉換。


- 建立完整的虛擬變數
我們本來處理類別變數的方式是單純將類別轉至數值，轉成虛擬變數（或One-hot encoding）之後，修正成有n種類別就新增n-1個欄位。

![img](https://i.imgur.com/mtimFxh.png)
- 建立更多變數
例如：我們將所以可居住面積加總，新增一個全新的變數**TotalSF**，方便我們更容易學習。

**特徵工程修改完成後總資料共有220變數以及2917個觀察值。**


# 二、 模型改進

原先使用隨機森林模型效果似乎還可以再進步，我們決定採用較複雜的模型組合。

## 混合模型


- KRR (Kernel Ridge Regression)
與SVR類似，皆使用L2範數正則化處理，但損失函數不同，比SVR運算更快速的學習法。在機器學習演算法中，要把特徵映射到高維空間再算內積，運算量會非常繁瑣，kernal是一個簡便實現這個過程的捷徑。


- Lasso
比傳統回歸多了L1範數的正則化處理，讓模型整體複雜程度不會太高，也可以自己把不重要的變數的係數降至為，複雜度調整的程度由參數λ來控制，λ越大對變量較多的線性模型的懲罰力度就越大

- ENet
是結合lasso和Ridge的迴歸

- GBoost
利用Boosting的方法來擬合模型，所謂的boosting就是根據上一次模型的錯誤再進行學習，藉此迭代下去。

- XGB
陳天奇教授提出的新的gradient boosting演算法，比傳統的Gboost訓練速度更高，準確度也有提升，是Kaggle上的大殺器。

- LGB
XGboost的決策樹選擇分割點，需要遍歷整個特徵值，但是LGB則是把特徵值形成一個直方圖的概念來提取，所以速度又更快了

最後將預測結果依以下權重組合：

        (KRR + ENet + GBoost + Lasso) * 0.3 + LGB * 0.4 + XGB * 0.3)

L1 L2區別總結:
加入正則項是為了避免過擬合,或解進行某種約束,需要保持某種特性 
L1正則可以保證模型的稀疏性，也就是某些參數等於0

L2正則可以保證模型的穩定性，也就是參數的值不會太大或太小，我們讓L2範數的正則項最小，可以使參數向量W的每個元素都很小，都接近於0。

但與L1範數不一樣的是，它不會是每個元素為0，而只是接近於0。越小的參數說明模型越簡單，越簡單的模型越不容易產生過擬合現象。


# 三、 最佳Kaggle結果

![img](https://i.imgur.com/l9HAXHf.png)

# 四、 結論
經過我們的分析後的變數重要程度列表如圖：
![圖](https://i.imgur.com/r5Fte0X.png)
可以看出**LotArea**(占地面積)以及**TotalSF**(總可用面積)表現得十分顯著，也相當符合經濟直覺。

我們認為改進過後的成績能有大幅度的進步主要來自下列原因：

- 變數的保留
原先以為刪除有過多缺值的變數會有優化模型的效果，結果可能因此將重要的資訊刪除。

- 模型的參數
經過敏感性分析調整參數，找出最佳成績的配適組合。

---
# 參考資料
[詳解 Kaggle 房價預測競賽優勝方案：用 Python 進行全面數據探索](https://www.hksilicon.com/articles/1316888)

[Kaggle实战（2）—房价预测：高阶回归技术应用](https://zhuanlan.zhihu.com/p/30415389)

[Comprehensive data exploration with Python by Pedro Marcelino](https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python)    

[Stacked Regressions : Top 4% on LeaderBoard by Serigne](https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard)
