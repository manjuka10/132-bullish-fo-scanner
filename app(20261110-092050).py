import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="1-3-2 Bullish F&O Scanner", layout="wide")
st.title("📈 1-3-2 Bullish F&O Scanner")
st.caption("Daily swing-trend scanner • Research tool only — not investment advice or a trade signal")

# Editable universe. Update this list against the current NSE F&O contracts list.
DEFAULT_SYMBOLS = """RELIANCE
HDFCBANK
ICICIBANK
SBIN
AXISBANK
KOTAKBANK
INDUSINDBK
BANKBARODA
PNB
FEDERALBNK
CANBK
IDFCFIRSTB
NIFTY
TATAMOTORS
MARUTI
M&M
BAJAJ-AUTO
HEROMOTOCO
EICHERMOT
TVSMOTOR
TATASTEEL
JSWSTEEL
HINDALCO
VEDL
COALINDIA
ONGC
IOC
BPCL
NTPC
POWERGRID
TATAPOWER
ADANIENT
ADANIPORTS
DLF
GODREJPROP
OBEROIRLTY
LODHA
LT
BEL
HAL
BHEL
CUMMINSIND
SIEMENS
ABB
AMBUJACEM
ULTRACEMCO
GRASIM
SHREECEM
ASIANPAINT
TITAN
TRENT
DMART
ITC
HINDUNILVR
NESTLEIND
BRITANNIA
TATACONSUM
DABUR
MARICO
SUNPHARMA
DRREDDY
CIPLA
DIVISLAB
LUPIN
AUROPHARMA
APOLLOHOSP
MAXHEALTH
FORTIS
LTIM
TCS
INFY
HCLTECH
WIPRO
TECHM
COFORGE
PERSISTENT
MPHASIS
OFSS
BHARTIARTL
IDEA
INDUSTOWER
ZOMATO
PAYTM
POLICYBZR
NYKAA
JIOFIN
CHOLAFIN
BAJFINANCE
BAJAJFINSV
SHRIRAMFIN
MUTHOOTFIN
PFC
RECLTD
IRFC
IREDA
"""
with st.sidebar:
    st.header("Scanner settings")
    symbols_text=st.text_area("Symbols (one NSE symbol per line)", DEFAULT_SYMBOLS, height=300)
    period=st.selectbox("History",["1y","2y","3y"],index=1)
    min_score=st.slider("Minimum bullish score",0,100,60,5)
    workers=st.slider("Parallel downloads",1,12,6)
    run=st.button("🔎 Scan stocks",type="primary",use_container_width=True)
    st.markdown("**Data:** Yahoo Finance daily candles (`.NS`), may be delayed/incomplete. Confirm prices and contracts in Kite/NSE.")

def calc_rsi(s, n=14):
    delta=s.diff()
    gain=delta.clip(lower=0).ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    loss=(-delta.clip(upper=0)).ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    rs=gain/loss.replace(0,np.nan)
    return 100-(100/(1+rs))

def calc_adx(df,n=14):
    h,l,c=df["High"],df["Low"],df["Close"]
    up=h.diff()
    down=-l.diff()
    plus_dm=pd.Series(np.where((up>down)&(up>0),up,0),index=df.index)
    minus_dm=pd.Series(np.where((down>up)&(down>0),down,0),index=df.index)
    tr=pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/n,min_periods=n,adjust=False).mean()
    pdi=100*plus_dm.ewm(alpha=1/n,min_periods=n,adjust=False).mean()/atr
    mdi=100*minus_dm.ewm(alpha=1/n,min_periods=n,adjust=False).mean()/atr
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    adx=dx.ewm(alpha=1/n,min_periods=n,adjust=False).mean()
    return adx,pdi,mdi,atr

def analyze(symbol, benchmark):
    ticker=symbol.strip().upper()
    if not ticker: return None
    data=yf.download(ticker+".NS",period=period,interval="1d",auto_adjust=True,progress=False,threads=False)
    if data.empty or len(data)<210: return {"Symbol":ticker,"Error":"Insufficient data"}
    if isinstance(data.columns,pd.MultiIndex): data.columns=data.columns.get_level_values(0)
    data=data.dropna()
    c=data["Close"]; v=data["Volume"]
    data["EMA20"]=c.ewm(span=20,adjust=False).mean()
    data["EMA50"]=c.ewm(span=50,adjust=False).mean()
    data["EMA200"]=c.ewm(span=200,adjust=False).mean()
    data["RSI"]=calc_rsi(c)
    data["ADX"],data["+DI"],data["-DI"],data["ATR"]=calc_adx(data)
    last=data.iloc[-1]; prev=data.iloc[-2]
    # relative strength over 20 sessions vs Nifty 50
    stock_ret=(c.iloc[-1]/c.iloc[-21]-1)*100 if len(c)>21 else np.nan
    nifty_ret=(benchmark.iloc[-1]/benchmark.iloc[-21]-1)*100 if len(benchmark)>21 else np.nan
    rs=stock_ret-nifty_ret
    vol_avg=v.rolling(20).mean().iloc[-1]
    vol_ratio=(last["Volume"]/vol_avg) if vol_avg and vol_avg>0 else np.nan
    resistance=data["High"].iloc[-21:-1].max()
    breakout=last["Close"]>resistance
    trend=(last["Close"]>last["EMA20"]>last["EMA50"]>last["EMA200"])
    ema20_rising=last["EMA20"]>data["EMA20"].iloc[-6]
    # Score components, max 100
    trend_score=25 if trend else (15 if last["Close"]>last["EMA50"]>last["EMA200"] else (8 if last["Close"]>last["EMA200"] else 0))
    adx_score=20 if last["ADX"]>=25 and last["+DI"]>last["-DI"] else (12 if last["+DI"]>last["-DI"] and last["ADX"]>=18 else 0)
    rs_score=20 if rs>=3 else (15 if rs>=1 else (8 if rs>0 else 0))
    breakout_score=15 if breakout else (8 if last["Close"]>resistance*0.98 else 0)
    volume_score=10 if vol_ratio>=1.5 else (6 if vol_ratio>=1.1 else 0)
    rsi_score=10 if 55<=last["RSI"]<=70 else (6 if 50<=last["RSI"]<55 or 70<last["RSI"]<=75 else 0)
    score=trend_score+adx_score+rs_score+breakout_score+volume_score+rsi_score
    if last["RSI"]>75: status="Extended"
    elif score>=80: status="Strong bullish"
    elif score>=65: status="Bullish"
    elif score>=min_score: status="Watch"
    else: status="Low score"
    return {"Symbol":ticker,"Price":round(float(last["Close"]),2),"Score":score,"Signal":status,
            "Trend":"Up" if trend else ("Above 200 EMA" if last["Close"]>last["EMA200"] else "Weak"),
            "EMA20 rising":"Yes" if ema20_rising else "No","ADX":round(float(last["ADX"]),1),
            "+DI":round(float(last["+DI"]),1),"-DI":round(float(last["-DI"]),1),
            "RSI":round(float(last["RSI"]),1),"RS vs Nifty (%)":round(float(rs),2),
            "Volume/20D":round(float(vol_ratio),2) if pd.notna(vol_ratio) else np.nan,
            "Breakout":"Yes" if breakout else "No",
            "ATR%":round(float(last["ATR"]/last["Close"]*100),2),
            "Resistance":round(float(resistance),2)}

if run:
    symbols=list(dict.fromkeys(s.strip().upper() for s in symbols_text.splitlines() if s.strip()))
    if not symbols: st.error("Enter at least one symbol.")
    else:
        with st.spinner("Downloading daily candles and calculating indicators..."):
            b=yf.download("^NSEI",period=period,interval="1d",auto_adjust=True,progress=False)
            if isinstance(b.columns,pd.MultiIndex): b.columns=b.columns.get_level_values(0)
            benchmark=b["Close"].dropna() if not b.empty else pd.Series(dtype=float)
            if benchmark.empty or len(benchmark)<30:
                st.error("Could not download Nifty benchmark data. Try again later.")
            else:
                rows=[]
                progress=st.progress(0)
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futures={ex.submit(analyze,s,benchmark):s for s in symbols}
                    for i,f in enumerate(as_completed(futures),1):
                        try:
                            r=f.result()
                            if r: rows.append(r)
                        except Exception as e: rows.append({"Symbol":futures[f],"Error":str(e)[:100]})
                        progress.progress(i/len(futures))
                result=pd.DataFrame(rows)
                if "Score" in result:
                    result=result.sort_values("Score",ascending=False,na_position="last")
                st.session_state["scan_result"]=result
                st.session_state["scan_date"]=date.today().isoformat()

if "scan_result" in st.session_state:
    result=st.session_state["scan_result"]
    st.caption(f"Last scan: {st.session_state.get('scan_date','')} • {len(result)} symbols")
    if "Score" in result:
        filtered=result[result["Score"].fillna(-1)>=min_score].copy()
        st.subheader("Bullish candidates")
        st.dataframe(filtered,hide_index=True,use_container_width=True)
        st.download_button("Download scan CSV",filtered.to_csv(index=False).encode("utf-8"),"132_bullish_scan.csv","text/csv")
        errs=result[result.get("Error",pd.Series(index=result.index,dtype=object)).notna()] if "Error" in result else pd.DataFrame()
        if not errs.empty:
            with st.expander(f"Symbols with data errors ({len(errs)})"): st.dataframe(errs,use_container_width=True)
        st.subheader("Score distribution")
        st.bar_chart(filtered.set_index("Symbol")["Score"] if not filtered.empty else pd.Series(dtype=float))
    else: st.dataframe(result,use_container_width=True)

st.divider()
st.markdown("""
### How to use this with your monthly 1-3-2 strategy
1. Scan after market close; shortlist **Strong bullish/Bullish** names.
2. Prefer price above rising EMA20/50/200, ADX above 20–25 with +DI > −DI, and positive relative strength.
3. Check liquidity, option bid–ask spread, OI, upcoming results/events, and chart levels in Kite.
4. Validate the option structure, max loss, margin, and adjustment/exit rules before entry.

**Important:** A high score is a technical filter, not a probability of profit. Backtest the exact 1-3-2 rules and paper-trade first. Options can lose substantial money.
""")
