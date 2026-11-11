import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests, gzip, io, re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="1-3-2 Bullish F&O Scanner", layout="wide")
st.title("📈 1-3-2 Bullish F&O Scanner")
st.caption("Automatic NSE individual-stock F&O universe • Daily technical screening")

NSE_HEADERS = {
 "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/125 Safari/537.36",
 "Accept":"*/*","Referer":"https://www.nseindia.com/"
}
CACHE_FILE="fo_symbols_cache.csv"
IST = ZoneInfo("Asia/Kolkata")
def ist_now(): return datetime.now(IST)

def get_nse_fo_symbols():
    """Fetch current NSE F&O underlyings via NSE's underlying API; use FYERS FO master fallback."""
    errors=[]
    # NSE's public underlying-information endpoint (JSON)
    nse_urls=[
      "https://www.nseindia.com/api/underlying-information",
      "https://www.nseindia.com/api/underlying-information?type=equity"
    ]
    headers={
      "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131 Safari/537.36",
      "Accept":"application/json,text/plain,*/*",
      "Referer":"https://www.nseindia.com/"
    }
    session=requests.Session(); session.headers.update(headers)
    try: session.get("https://www.nseindia.com",timeout=10)
    except Exception: pass
    for url in nse_urls:
        try:
            r=session.get(url,timeout=20); r.raise_for_status()
            j=r.json()
            # NSE response keys can change; recursively collect likely symbol fields.
            found=set()
            def walk(obj):
                if isinstance(obj,dict):
                    for k,v in obj.items():
                        if str(k).lower() in {"symbol","underlying","underlyingsymbol","underlying_symbol"} and isinstance(v,str):
                            s=v.strip().upper()
                            if re.fullmatch(r"[A-Z0-9&\-]+",s): found.add(s)
                        walk(v)
                elif isinstance(obj,list):
                    for v in obj: walk(v)
            walk(j)
            found-= {"NIFTY","BANKNIFTY","FINNIFTY","MIDCPNIFTY","NIFTYNXT50"}
            if len(found)>=100:
                syms=sorted(found); pd.DataFrame({"Symbol":syms}).to_csv(CACHE_FILE,index=False)
                return syms,ist_now().strftime("%d-%b-%Y"),"NSE underlying-information API"
        except Exception as e: errors.append("NSE API: "+str(e)[:150])
    # Public FYERS NSE equity-derivatives symbol master fallback
    try:
        fy="https://public.fyers.in/sym_details/NSE_FO.csv"
        r=requests.get(fy,headers={"User-Agent":"Mozilla/5.0"},timeout=35); r.raise_for_status()
        raw=pd.read_csv(io.BytesIO(r.content),header=None,low_memory=False)
        # Symbol Details typically contains NSE:SYMBOL...; identify stock underlyings
        details=raw.astype(str).agg(" ".join,axis=1).str.upper()
        found=set()
        for text in details:
            for match in re.findall(r"(?:NSE:)?([A-Z][A-Z0-9&\-]{1,20})(?:[-_](?:FUT|CE|PE)|\d{2}[A-Z]{3}\d{2,4}|[-_]\d)",text):
                if match not in {"NIFTY","BANKNIFTY","FINNIFTY","MIDCPNIFTY","NIFTYNXT50"}: found.add(match)
        if len(found)>=100:
            syms=sorted(found); pd.DataFrame({"Symbol":syms}).to_csv(CACHE_FILE,index=False)
            return syms,ist_now().strftime("%d-%b-%Y"),"FYERS NSE_FO symbol master"
        errors.append("FYERS master: parsed only "+str(len(found))+" symbols")
    except Exception as e: errors.append("FYERS master: "+str(e)[:150])
    # Previous successful download only; do not hardcode/manual universe.
    try:
        old=pd.read_csv(CACHE_FILE)["Symbol"].dropna().astype(str).str.upper().unique().tolist()
        if len(old)>=100: return old,"Previous successful fetch","Cached F&O list (may be stale)"
    except Exception: pass
    raise RuntimeError("Unable to retrieve current F&O universe. "+" | ".join(errors))

def rsi(s,n=14):
    delta=s.diff()
    gain=delta.clip(lower=0).ewm(alpha=1/n,min_periods=n,adjust=False).mean()
    loss=(-delta.clip(upper=0)).ewm(alpha=1/n,min_periods=n,adjust=False).mean()
    return 100-100/(1+gain/loss.replace(0,np.nan))

def adx_calc(df,n=14):
    h,l,c=df.High,df.Low,df.Close
    up=h.diff(); down=-l.diff()
    pdm=pd.Series(np.where((up>down)&(up>0),up,0),index=df.index)
    mdm=pd.Series(np.where((down>up)&(down>0),down,0),index=df.index)
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/n,min_periods=n,adjust=False).mean()
    pdi=100*pdm.ewm(alpha=1/n,min_periods=n,adjust=False).mean()/atr
    mdi=100*mdm.ewm(alpha=1/n,min_periods=n,adjust=False).mean()/atr
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    return dx.ewm(alpha=1/n,min_periods=n,adjust=False).mean(),pdi,mdi,atr

def scan_one(sym,bench,period):
    try:
        df=yf.download(sym+".NS",period=period,interval="1d",auto_adjust=True,progress=False,threads=False)
        if df.empty: return {"Symbol":sym,"Error":"No Yahoo data"}
        if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
        df=df.dropna()
        if len(df)<205: return {"Symbol":sym,"Error":"Insufficient history"}
        c=df.Close; v=df.Volume
        df["E20"]=c.ewm(span=20,adjust=False).mean(); df["E50"]=c.ewm(span=50,adjust=False).mean(); df["E200"]=c.ewm(span=200,adjust=False).mean()
        df["RSI"]=rsi(c); df["ADX"],df["PDI"],df["MDI"],df["ATR"]=adx_calc(df)
        x=df.iloc[-1]
        stock20=(c.iloc[-1]/c.iloc[-21]-1)*100
        nifty20=(bench.iloc[-1]/bench.iloc[-21]-1)*100
        rs=stock20-nifty20
        avgvol=v.rolling(20).mean().iloc[-1]
        vr=x.Volume/avgvol if avgvol and avgvol>0 else np.nan
        # Prior 20-session breakout reference (excluding today's candle)
        breakout_res=float(df.High.iloc[-21:-1].max())
        # Nearest overhead resistance: local swing highs from recent 120 sessions above current close
        highs=df.High.iloc[-121:-1].copy()
        swing_mask=(highs.shift(1)<highs) & (highs.shift(-1)<=highs)
        swing_levels=highs[swing_mask]
        overhead=swing_levels[swing_levels>x.Close]
        next_res=float(overhead.min()) if not overhead.empty else np.nan
        trend=x.Close>x.E20>x.E50>x.E200
        rising=x.E20>df.E20.iloc[-6]
        breakout=x.Close>breakout_res
        s_tr=25 if trend else (15 if x.Close>x.E50>x.E200 else (8 if x.Close>x.E200 else 0))
        s_adx=20 if x.ADX>=25 and x.PDI>x.MDI else (12 if x.PDI>x.MDI and x.ADX>=18 else 0)
        s_rs=20 if rs>=3 else (15 if rs>=1 else (8 if rs>0 else 0))
        s_br=15 if breakout else (8 if x.Close>breakout_res*.98 else 0)
        s_vol=10 if vr>=1.5 else (6 if vr>=1.1 else 0)
        s_rsi=10 if 55<=x.RSI<=70 else (6 if 50<=x.RSI<55 or 70<x.RSI<=75 else 0)
        score=s_tr+s_adx+s_rs+s_br+s_vol+s_rsi
        signal="Extended" if x.RSI>75 else ("Strong bullish" if score>=80 else "Bullish" if score>=65 else "Watch" if score>=55 else "Low score")
        return {"Symbol":sym,"Price":round(float(x.Close),2),"Score":score,"Signal":signal,
          "Trend":"EMA aligned" if trend else "Not aligned","EMA20 rising":"Yes" if rising else "No",
          "ADX":round(float(x.ADX),1),"+DI":round(float(x.PDI),1),"-DI":round(float(x.MDI),1),
          "RSI":round(float(x.RSI),1),"RS vs Nifty (%)":round(float(rs),2),
          "Volume/20D":round(float(vr),2) if pd.notna(vr) else np.nan,
          "Breakout":"Yes" if breakout else "No","ATR%":round(float(x.ATR/x.Close*100),2),
          "20D Breakout Level":round(float(breakout_res),2),
          "Next Resistance Above":round(float(next_res),2) if pd.notna(next_res) else "Not found"}
    except Exception as e: return {"Symbol":sym,"Error":str(e)[:120]}

# Top controls
a,b,c=st.columns([1,1,2])
with a: period=st.selectbox("History",["1y","2y","3y"],index=1)
with b: minimum=st.selectbox("Minimum score",[50,55,60,65,70,75,80],index=2)
with c: workers=st.slider("Parallel downloads",2,12,6)
refresh=st.button("🔄 Refresh NSE F&O list + Scan",type="primary",use_container_width=True)

if refresh:
    try:
        with st.spinner("Fetching current NSE F&O contract universe..."):
            symbols,fo_date,source=get_nse_fo_symbols()
        st.session_state["symbols"]=symbols
        st.session_state["fo_date"]=fo_date
        st.session_state["fo_source"]=source
        st.session_state["fo_updated"]=ist_now().strftime("%d-%b-%Y %I:%M:%S %p IST")
        st.success(f"Loaded {len(symbols)} NSE F&O stocks • {source}")
        with st.spinner(f"Scanning {len(symbols)} stocks using daily data..."):
            bd=yf.download("^NSEI",period=period,interval="1d",auto_adjust=True,progress=False)
            if isinstance(bd.columns,pd.MultiIndex): bd.columns=bd.columns.get_level_values(0)
            bench=bd.Close.dropna()
            rows=[]
            progress=st.progress(0)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                fs={pool.submit(scan_one,s,bench,period):s for s in symbols}
                for i,f in enumerate(as_completed(fs),1):
                    rows.append(f.result()); progress.progress(i/len(fs))
            res=pd.DataFrame(rows)
            if "Score" in res: res=res.sort_values("Score",ascending=False,na_position="last")
            st.session_state["results"]=res
            st.session_state["market_updated"]=ist_now().strftime("%d-%b-%Y %I:%M:%S %p IST")
    except Exception as e: st.error(f"Scan failed: {e}")

# Status / results
if st.session_state.get("symbols"):
    st.info(f"F&O universe: {len(st.session_state.symbols)} stocks | NSE contract date: {st.session_state.get('fo_date')} | List fetched: {st.session_state.get('fo_updated')} | Source: {st.session_state.get('fo_source')}")
if st.session_state.get("market_updated"): st.caption("Market data scan completed: "+st.session_state.market_updated)
if "results" in st.session_state:
    df=st.session_state.results
    if "Score" in df:
        candidates=df[df.Score.fillna(-1)>=minimum]
        st.subheader(f"Bullish candidates ({len(candidates)})")
        st.dataframe(candidates,hide_index=True,use_container_width=True)
        st.download_button("⬇️ Download candidates CSV",candidates.to_csv(index=False).encode(),"132_bullish_candidates.csv","text/csv")
        with st.expander("All scanned stocks / data errors"):
            st.dataframe(df,hide_index=True,use_container_width=True)
    else: st.dataframe(df,use_container_width=True)

st.markdown("""
### Monthly 1-3-2 workflow
- Scan after market close; review Strong bullish/Bullish candidates.
- Validate trend, relative strength, liquidity, option bid–ask, OI, event risk and chart levels in Kite.
- Check exact 1-3-2 payoff, margin, maximum loss, adjustment and exit plan before entry.

**Caution:** NSE can block cloud requests. If the official list download fails, app shows an error rather than silently using a hard-coded universe. Yahoo Finance daily data may be delayed/incomplete. A score is a filter—not probability of profit. Backtest and paper-trade first.
""")
