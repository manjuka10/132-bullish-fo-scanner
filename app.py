import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests, gzip, io, re
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="1-3-2 Bullish F&O Scanner", layout="wide")
st.title("📈 1-3-2 Bullish F&O Scanner")
st.caption("Automatic NSE individual-stock F&O universe • Daily technical screening")

NSE_HEADERS = {
 "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/125 Safari/537.36",
 "Accept":"*/*","Referer":"https://www.nseindia.com/"
}
CACHE_FILE="fo_symbols_cache.csv"

def get_nse_fo_symbols():
    """Try current dated NSE MII contract gzip; derive stock underlyings from FUTSTK/OPTSTK."""
    session=requests.Session()
    session.headers.update(NSE_HEADERS)
    # Establish NSE cookies; some deployments may block this request.
    try: session.get("https://www.nseindia.com",timeout=12)
    except Exception: pass
    today=datetime.now().date()
    errors=[]
    for days_back in range(0,8):
        d=today-timedelta(days=days_back)
        date_tag=d.strftime("%d%m%Y")
        urls=[
          f"https://nsearchives.nseindia.com/content/fo/NSE_FO_contract_{date_tag}.csv.gz",
          f"https://archives.nseindia.com/content/fo/NSE_FO_contract_{date_tag}.csv.gz",
          f"https://www.nseindia.com/content/fo/NSE_FO_contract_{date_tag}.csv.gz"
        ]
        for url in urls:
            try:
                r=session.get(url,timeout=25)
                if r.status_code!=200 or len(r.content)<1000: continue
                raw=gzip.decompress(r.content) if r.content[:2]==b"\\x1f\\x8b" else r.content
                df=pd.read_csv(io.BytesIO(raw),low_memory=False)
                cols={str(c).strip().upper():c for c in df.columns}
                # Contract file field names vary by NSE version.
                inst_col=next((cols[k] for k in ["INSTRUMENT","INSTRUMENTTYPE","INSTRUMENT_TYPE"] if k in cols),None)
                sym_col=next((cols[k] for k in ["SYMBOL","UNDERLYING","UNDERLYING_SYMBOL"] if k in cols),None)
                if not inst_col or not sym_col: continue
                inst=df[inst_col].astype(str).str.upper()
                symbols=df.loc[inst.isin(["FUTSTK","OPTSTK"]),sym_col].astype(str).str.strip().str.upper()
                symbols=symbols[symbols.str.match(r"^[A-Z0-9&\-]+$")].drop_duplicates().sort_values()
                if len(symbols)>100:
                    out=pd.DataFrame({"Symbol":symbols})
                    out.to_csv(CACHE_FILE,index=False)
                    return list(symbols),d.strftime("%d-%b-%Y"),"NSE live contract file"
            except Exception as e: errors.append(str(e))
    # Fallback only if official request is unavailable; explicitly label cached list.
    try:
        cached=pd.read_csv(CACHE_FILE)["Symbol"].dropna().astype(str).str.upper().unique().tolist()
        if len(cached)>100: return cached,"Cache date unknown","Cached NSE universe (verify freshness)"
    except Exception: pass
    raise RuntimeError("NSE contract file could not be downloaded and no saved F&O universe exists. NSE may block Streamlit Cloud. Please retry later.")

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
        res=df.High.iloc[-21:-1].max()
        trend=x.Close>x.E20>x.E50>x.E200
        rising=x.E20>df.E20.iloc[-6]
        breakout=x.Close>res
        s_tr=25 if trend else (15 if x.Close>x.E50>x.E200 else (8 if x.Close>x.E200 else 0))
        s_adx=20 if x.ADX>=25 and x.PDI>x.MDI else (12 if x.PDI>x.MDI and x.ADX>=18 else 0)
        s_rs=20 if rs>=3 else (15 if rs>=1 else (8 if rs>0 else 0))
        s_br=15 if breakout else (8 if x.Close>res*.98 else 0)
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
          "Resistance":round(float(res),2)}
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
        st.session_state["fo_updated"]=datetime.now().astimezone().strftime("%d-%b-%Y %I:%M:%S %p %Z")
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
            st.session_state["market_updated"]=datetime.now().astimezone().strftime("%d-%b-%Y %I:%M:%S %p %Z")
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
