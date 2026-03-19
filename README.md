# Crypto Perdiction Script
a simple script to predict crypto prices
# Setup
- You must install python LTS 3.10 or greater
- Get a [Tiingo api key](https://www.tiingo.com/documentation/general/overview)
- Set your api key in env vars before running the script
- ``` export TIINGO_API_KEY="<set_your_api_key>"; ```
- ``` bash pip install -r requirements.txt  ```
- Run the script
- ```python crypto_predict.py --source tiingo --top 5```

## Usage
---
### Analyze specific coins with more history
```python crypto_predict.py --source tiingo --days 1095 --coins BTC ETH SOL ADA DOGE AVAX```
### Show only the top 5 ranked
```python crypto_predict.py --source tiingo --top 5 ```
### Extend lookback window for more training data
```python crypto_predict.py --source tiingo --days 1095 ```
### Output:
<img width="820" height="825" alt="image" src="https://github.com/user-attachments/assets/82d21f55-bd82-4aa2-80cf-cdbf1193e713" />


---
# Prediction Method Overview
Given everything we know about this coin's price behavior today, is it more likely to be higher or lower 7 days from now?

## Step 1: Label the past
For every historical day in the training data, we look 7 days into the future and mark it 1 (price went up) or 0 (price went down). This becomes the "ground truth" the model learns from. So we're not predicting how much it moves - just which direction.

## Step 2: Turn price history into features
Raw prices aren't useful to a model directly - "BTC was $82,000" means nothing without context. So we engineer 20 features that capture market conditions on each day:
*Every row in the dataset is now a snapshot: "on this day, these were the 20 market conditions, and the coin went UP/DOWN 7 days later."*

- **Momentum** (did it recently trend up or down?) - the 1/3/7/14/30-day return columns
**Trend vs. moving averages** - is price above or below its 7/14/21/50-day average? A coin trading above its 50-day SMA is in a different regime than one below it
- **RSI** - measures if it's overbought (>70) or oversold (<30), classic mean-reversion signal
- **MACD** - compares two exponential moving averages; the histogram captures momentum acceleration
- **Bollinger Band** % - where is price sitting within its recent volatility range? Near the top vs. bottom tells the model something
- **Volatility** - high vol markets behave differently than low vol ones
- **Volume signals** - unusual volume often precedes price moves



## Step 3: Train a Gradient Boosting classifier
- This is where the actual learning happens. Gradient Boosting builds an ensemble of decision trees sequentially, where each tree tries to correct the errors of the previous ones.
- Think of it like this: imagine you had a panel of 200 analysts, where each one specializes in catching the mistakes the previous 199 made. Their combined vote is the final prediction. Each individual "analyst" (tree) is deliberately kept shallow (max depth 4) so no single one overfits to noise - the signal only emerges from their ensemble.
The model learns things like: "when RSI is below 35 AND price is 15% below the 50-day SMA AND volume spiked yesterday, coins went up 7 days later 65% of the time in the training data." It discovers thousands of these patterns automatically.

## Step 4: Time-series cross-validation (the honest part)
- This is the most important and subtle step. We can't use regular random cross-validation because the future can't train on the past. If you randomly shuffle the data and train on some of it, you'll accidentally give the model future information to predict the past - it'll look great in testing but fail in live use.
Instead we use TimeSeriesSplit with 5 folds, which works like  below:
    - The "CV Acc" you see in the output (typically 54-58%) is this honest estimate. 
    - It's intentionally modest - if you saw 75%+ something is wrong.
 ```
(figure 1)
# always testing on data the model has never seen
Fold 1: Train on months 1-4   → Test on month 5
Fold 2: Train on months 1-8   → Test on month 9
Fold 3: Train on months 1-12  → Test on month 13
```   

## Step 5: Predict on today
- After CV validation, we retrain on all historical data except the last 7 days (since we don't yet know if those went up), then run today's 20 feature values through the trained model.
- The output isn't a binary yes/no - it's predict_proba(), which gives a probability between 0 and 1. A score of 0.67 means "in situations historically similar to today's market conditions, this coin went up 67% of the time over the next 7 days."
- The ranking is then just sorting all coins by that probability, highest to lowest.

### What the model is actually good at vs. not:
- It's good at capturing regime signals - things like oversold bounces, momentum continuation, and volatility contractions that have historically repeated across crypto markets.
- It's bad at black swans - a regulatory announcement, an exchange collapse, a macro shock. No amount of RSI or MACD will predict those. The 54-58% accuracy reflects a model that's learned genuine signal, but crypto is noisy enough that roughly 45% of its "up" calls will still be wrong. The value is in the ranking across coins on any given day, not in treating any single prediction as a certainty.
