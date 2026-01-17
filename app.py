#!/usr/bin/env python3
"""S&P 500 Mean Reversion Analysis - Top 10 stocks likely to revert to mean within 30 days"""

import numpy as np
from scipy import stats
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict
import logging, threading, os
from flask import Flask, jsonify

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

stock_data_cache: Dict[str, pd.DataFrame] = {}
fetch_status = {"in_progress": False, "completed": 0, "failed": 0, "total": 0, "message": "", "last_fetch": None}

SP500 = ["A","AAPL","ABBV","ABNB","ABT","ACGL","ACN","ADBE","ADI","ADM","ADP","ADSK","AEE","AEP","AES","AFL","AIG","AIZ","AJG","AKAM","ALB","ALGN","ALL","ALLE","AMAT","AMCR","AMD","AME","AMGN","AMP","AMT","AMZN","ANET","ANSS","AON","AOS","APA","APD","APH","APTV","ARE","ATO","AVB","AVGO","AVY","AWK","AXON","AXP","AZO","BA","BAC","BALL","BAX","BBWI","BBY","BDX","BEN","BG","BIIB","BIO","BK","BKNG","BKR","BLDR","BLK","BMY","BR","BRK-B","BRO","BSX","BWA","BX","BXP","C","CAG","CAH","CARR","CAT","CB","CBOE","CBRE","CCI","CCL","CDNS","CDW","CE","CEG","CF","CFG","CHD","CHRW","CHTR","CI","CINF","CL","CLX","CMA","CMCSA","CME","CMG","CMI","CMS","CNC","CNP","COF","COO","COP","COR","COST","CPAY","CPB","CPRT","CPT","CRL","CRM","CSCO","CSGP","CSX","CTAS","CTLT","CTRA","CTSH","CTVA","CVS","CVX","D","DAL","DAY","DD","DE","DECK","DFS","DG","DGX","DHI","DHR","DIS","DLR","DLTR","DOC","DOV","DOW","DPZ","DRI","DTE","DUK","DVA","DVN","DXCM","EA","EBAY","ECL","ED","EFX","EG","EIX","EL","ELV","EMN","EMR","ENPH","EOG","EPAM","EQIX","EQR","EQT","ES","ESS","ETN","ETR","ETSY","EVRG","EW","EXC","EXPD","EXPE","EXR","F","FANG","FAST","FCX","FDS","FDX","FE","FFIV","FI","FICO","FIS","FITB","FLT","FMC","FOX","FOXA","FRT","FSLR","FTNT","FTV","GD","GDDY","GE","GEHC","GEN","GEV","GILD","GIS","GL","GLW","GM","GNRC","GOOG","GOOGL","GPC","GPN","GRMN","GS","GWW","HAL","HAS","HBAN","HCA","HD","HES","HIG","HII","HLT","HOLX","HON","HPE","HPQ","HRL","HSIC","HST","HSY","HUBB","HUM","HWM","IBM","ICE","IDXX","IEX","IFF","ILMN","INCY","INTC","INTU","INVH","IP","IPG","IQV","IR","IRM","ISRG","IT","ITW","J","JBHT","JBL","JCI","JKHY","JNJ","JNPR","JPM","K","KDP","KEY","KEYS","KHC","KIM","KKR","KLAC","KMB","KMI","KMX","KO","KR","KVUE","L","LDOS","LEN","LH","LHX","LIN","LKQ","LLY","LMT","LNT","LOW","LRCX","LULU","LUV","LVS","LW","LYB","LYV","MA","MAA","MAR","MAS","MCD","MCHP","MCK","MCO","MDLZ","MDT","MET","META","MGM","MHK","MKC","MKTX","MLM","MMC","MMM","MNST","MO","MOH","MOS","MPC","MPWR","MRK","MRNA","MRO","MS","MSCI","MSFT","MSI","MTB","MTCH","MTD","MU","NCLH","NDAQ","NDSN","NEE","NEM","NFLX","NI","NKE","NOC","NOW","NRG","NSC","NTAP","NTRS","NUE","NVDA","NVR","NWS","NWSA","O","ODFL","OKE","OMC","ON","ORCL","ORLY","OTIS","OXY","PANW","PARA","PAYC","PAYX","PCAR","PCG","PEG","PEP","PFE","PFG","PG","PGR","PH","PHM","PKG","PLD","PM","PNC","PNR","PNW","PODD","POOL","PPG","PPL","PRU","PSA","PSX","PTC","PWR","PXD","QCOM","QRVO","RCL","REG","REGN","RF","RJF","RL","RMD","ROK","ROL","ROP","ROST","RSG","RTX","SBAC","SBUX","SCHW","SHW","SJM","SLB","SMCI","SNA","SNPS","SO","SOLV","SPG","SPGI","SRE","STE","STLD","STT","STX","STZ","SWK","SWKS","SYF","SYK","SYY","T","TAP","TDG","TDY","TECH","TEL","TER","TFC","TFX","TGT","TJX","TMO","TMUS","TPR","TRGP","TRMB","TROW","TRV","TSCO","TSLA","TSN","TT","TTWO","TXN","TXT","TYL","UAL","UBER","UDR","UHS","ULTA","UNH","UNP","UPS","URI","USB","V","VICI","VLO","VLTO","VMC","VRSK","VRSN","VRTX","VST","VTR","VTRS","VZ","WAB","WAT","WBA","WBD","WDC","WEC","WELL","WFC","WM","WMB","WMT","WRB","WST","WTW","WY","WYNN","XEL","XOM","XYL","YUM","ZBH","ZBRA","ZTS"]

NAMES = {"A":"Agilent","AAPL":"Apple","ABBV":"AbbVie","ABNB":"Airbnb","ABT":"Abbott","ACGL":"Arch Capital","ACN":"Accenture","ADBE":"Adobe","ADI":"Analog Devices","ADM":"ADM","ADP":"ADP","ADSK":"Autodesk","AEE":"Ameren","AEP":"AEP","AES":"AES","AFL":"Aflac","AIG":"AIG","AIZ":"Assurant","AJG":"Gallagher","AKAM":"Akamai","ALB":"Albemarle","ALGN":"Align Tech","ALL":"Allstate","ALLE":"Allegion","AMAT":"Applied Materials","AMCR":"Amcor","AMD":"AMD","AME":"AMETEK","AMGN":"Amgen","AMP":"Ameriprise","AMT":"American Tower","AMZN":"Amazon","ANET":"Arista","ANSS":"ANSYS","AON":"Aon","AOS":"A.O. Smith","APA":"APA","APD":"Air Products","APH":"Amphenol","APTV":"Aptiv","ARE":"Alexandria RE","ATO":"Atmos Energy","AVB":"AvalonBay","AVGO":"Broadcom","AVY":"Avery Dennison","AWK":"American Water","AXON":"Axon","AXP":"American Express","AZO":"AutoZone","BA":"Boeing","BAC":"Bank of America","BALL":"Ball Corp","BAX":"Baxter","BBWI":"Bath & Body Works","BBY":"Best Buy","BDX":"Becton Dickinson","BEN":"Franklin Resources","BG":"Bunge","BIIB":"Biogen","BIO":"Bio-Rad","BK":"BNY Mellon","BKNG":"Booking","BKR":"Baker Hughes","BLDR":"Builders FirstSource","BLK":"BlackRock","BMY":"Bristol-Myers","BR":"Broadridge","BRK-B":"Berkshire Hathaway","BRO":"Brown & Brown","BSX":"Boston Scientific","BWA":"BorgWarner","BX":"Blackstone","BXP":"Boston Properties","C":"Citigroup","CAG":"Conagra","CAH":"Cardinal Health","CARR":"Carrier","CAT":"Caterpillar","CB":"Chubb","CBOE":"Cboe","CBRE":"CBRE","CCI":"Crown Castle","CCL":"Carnival","CDNS":"Cadence","CDW":"CDW","CE":"Celanese","CEG":"Constellation Energy","CF":"CF Industries","CFG":"Citizens Financial","CHD":"Church & Dwight","CHRW":"C.H. Robinson","CHTR":"Charter","CI":"Cigna","CINF":"Cincinnati Financial","CL":"Colgate","CLX":"Clorox","CMA":"Comerica","CMCSA":"Comcast","CME":"CME Group","CMG":"Chipotle","CMI":"Cummins","CMS":"CMS Energy","CNC":"Centene","CNP":"CenterPoint","COF":"Capital One","COO":"Cooper","COP":"ConocoPhillips","COR":"Cencora","COST":"Costco","CPAY":"Corpay","CPB":"Campbell Soup","CPRT":"Copart","CPT":"Camden Property","CRL":"Charles River","CRM":"Salesforce","CSCO":"Cisco","CSGP":"CoStar","CSX":"CSX","CTAS":"Cintas","CTLT":"Catalent","CTRA":"Coterra","CTSH":"Cognizant","CTVA":"Corteva","CVS":"CVS","CVX":"Chevron","D":"Dominion","DAL":"Delta","DAY":"Dayforce","DD":"DuPont","DE":"Deere","DECK":"Deckers","DFS":"Discover","DG":"Dollar General","DGX":"Quest","DHI":"D.R. Horton","DHR":"Danaher","DIS":"Disney","DLR":"Digital Realty","DLTR":"Dollar Tree","DOC":"Healthpeak","DOV":"Dover","DOW":"Dow","DPZ":"Domino's","DRI":"Darden","DTE":"DTE Energy","DUK":"Duke Energy","DVA":"DaVita","DVN":"Devon","DXCM":"DexCom","EA":"EA","EBAY":"eBay","ECL":"Ecolab","ED":"Con Edison","EFX":"Equifax","EG":"Everest","EIX":"Edison Intl","EL":"Estee Lauder","ELV":"Elevance","EMN":"Eastman","EMR":"Emerson","ENPH":"Enphase","EOG":"EOG","EPAM":"EPAM","EQIX":"Equinix","EQR":"Equity Residential","EQT":"EQT","ES":"Eversource","ESS":"Essex Property","ETN":"Eaton","ETR":"Entergy","ETSY":"Etsy","EVRG":"Evergy","EW":"Edwards Life","EXC":"Exelon","EXPD":"Expeditors","EXPE":"Expedia","EXR":"Extra Space","F":"Ford","FANG":"Diamondback","FAST":"Fastenal","FCX":"Freeport","FDS":"FactSet","FDX":"FedEx","FE":"FirstEnergy","FFIV":"F5","FI":"Fiserv","FICO":"FICO","FIS":"FIS","FITB":"Fifth Third","FLT":"Fleetcor","FMC":"FMC","FOX":"Fox B","FOXA":"Fox A","FRT":"Federal Realty","FSLR":"First Solar","FTNT":"Fortinet","FTV":"Fortive","GD":"General Dynamics","GDDY":"GoDaddy","GE":"GE Aerospace","GEHC":"GE HealthCare","GEN":"Gen Digital","GEV":"GE Vernova","GILD":"Gilead","GIS":"General Mills","GL":"Globe Life","GLW":"Corning","GM":"GM","GNRC":"Generac","GOOG":"Alphabet C","GOOGL":"Alphabet A","GPC":"Genuine Parts","GPN":"Global Payments","GRMN":"Garmin","GS":"Goldman Sachs","GWW":"Grainger","HAL":"Halliburton","HAS":"Hasbro","HBAN":"Huntington","HCA":"HCA","HD":"Home Depot","HES":"Hess","HIG":"Hartford","HII":"Huntington Ingalls","HLT":"Hilton","HOLX":"Hologic","HON":"Honeywell","HPE":"HPE","HPQ":"HP","HRL":"Hormel","HSIC":"Henry Schein","HST":"Host Hotels","HSY":"Hershey","HUBB":"Hubbell","HUM":"Humana","HWM":"Howmet","IBM":"IBM","ICE":"ICE","IDXX":"IDEXX","IEX":"IDEX","IFF":"IFF","ILMN":"Illumina","INCY":"Incyte","INTC":"Intel","INTU":"Intuit","INVH":"Invitation Homes","IP":"Intl Paper","IPG":"IPG","IQV":"IQVIA","IR":"Ingersoll Rand","IRM":"Iron Mountain","ISRG":"Intuitive Surgical","IT":"Gartner","ITW":"ITW","J":"Jacobs","JBHT":"J.B. Hunt","JBL":"Jabil","JCI":"Johnson Controls","JKHY":"Jack Henry","JNJ":"J&J","JNPR":"Juniper","JPM":"JPMorgan","K":"Kellanova","KDP":"Keurig Dr Pepper","KEY":"KeyCorp","KEYS":"Keysight","KHC":"Kraft Heinz","KIM":"Kimco","KKR":"KKR","KLAC":"KLA","KMB":"Kimberly-Clark","KMI":"Kinder Morgan","KMX":"CarMax","KO":"Coca-Cola","KR":"Kroger","KVUE":"Kenvue","L":"Loews","LDOS":"Leidos","LEN":"Lennar","LH":"Labcorp","LHX":"L3Harris","LIN":"Linde","LKQ":"LKQ","LLY":"Eli Lilly","LMT":"Lockheed","LNT":"Alliant Energy","LOW":"Lowe's","LRCX":"Lam Research","LULU":"Lululemon","LUV":"Southwest","LVS":"Las Vegas Sands","LW":"Lamb Weston","LYB":"LyondellBasell","LYV":"Live Nation","MA":"Mastercard","MAA":"Mid-America Apt","MAR":"Marriott","MAS":"Masco","MCD":"McDonald's","MCHP":"Microchip","MCK":"McKesson","MCO":"Moody's","MDLZ":"Mondelez","MDT":"Medtronic","MET":"MetLife","META":"Meta","MGM":"MGM","MHK":"Mohawk","MKC":"McCormick","MKTX":"MarketAxess","MLM":"Martin Marietta","MMC":"Marsh McLennan","MMM":"3M","MNST":"Monster","MO":"Altria","MOH":"Molina","MOS":"Mosaic","MPC":"Marathon Petroleum","MPWR":"Monolithic Power","MRK":"Merck","MRNA":"Moderna","MRO":"Marathon Oil","MS":"Morgan Stanley","MSCI":"MSCI","MSFT":"Microsoft","MSI":"Motorola","MTB":"M&T Bank","MTCH":"Match","MTD":"Mettler-Toledo","MU":"Micron","NCLH":"Norwegian Cruise","NDAQ":"Nasdaq","NDSN":"Nordson","NEE":"NextEra","NEM":"Newmont","NFLX":"Netflix","NI":"NiSource","NKE":"Nike","NOC":"Northrop","NOW":"ServiceNow","NRG":"NRG","NSC":"Norfolk Southern","NTAP":"NetApp","NTRS":"Northern Trust","NUE":"Nucor","NVDA":"NVIDIA","NVR":"NVR","NWS":"News Corp B","NWSA":"News Corp A","O":"Realty Income","ODFL":"Old Dominion","OKE":"ONEOK","OMC":"Omnicom","ON":"ON Semi","ORCL":"Oracle","ORLY":"O'Reilly","OTIS":"Otis","OXY":"Occidental","PANW":"Palo Alto","PARA":"Paramount","PAYC":"Paycom","PAYX":"Paychex","PCAR":"PACCAR","PCG":"PG&E","PEG":"PSEG","PEP":"PepsiCo","PFE":"Pfizer","PFG":"Principal","PG":"P&G","PGR":"Progressive","PH":"Parker-Hannifin","PHM":"PulteGroup","PKG":"Packaging Corp","PLD":"Prologis","PM":"Philip Morris","PNC":"PNC","PNR":"Pentair","PNW":"Pinnacle West","PODD":"Insulet","POOL":"Pool Corp","PPG":"PPG","PPL":"PPL","PRU":"Prudential","PSA":"Public Storage","PSX":"Phillips 66","PTC":"PTC","PWR":"Quanta","PXD":"Pioneer","QCOM":"Qualcomm","QRVO":"Qorvo","RCL":"Royal Caribbean","REG":"Regency Centers","REGN":"Regeneron","RF":"Regions","RJF":"Raymond James","RL":"Ralph Lauren","RMD":"ResMed","ROK":"Rockwell","ROL":"Rollins","ROP":"Roper","ROST":"Ross","RSG":"Republic Services","RTX":"RTX","SBAC":"SBA Comm","SBUX":"Starbucks","SCHW":"Schwab","SHW":"Sherwin-Williams","SJM":"J.M. Smucker","SLB":"Schlumberger","SMCI":"Super Micro","SNA":"Snap-on","SNPS":"Synopsys","SO":"Southern Co","SOLV":"Solventum","SPG":"Simon Property","SPGI":"S&P Global","SRE":"Sempra","STE":"STERIS","STLD":"Steel Dynamics","STT":"State Street","STX":"Seagate","STZ":"Constellation Brands","SWK":"Stanley Black","SWKS":"Skyworks","SYF":"Synchrony","SYK":"Stryker","SYY":"Sysco","T":"AT&T","TAP":"Molson Coors","TDG":"TransDigm","TDY":"Teledyne","TECH":"Bio-Techne","TEL":"TE Connectivity","TER":"Teradyne","TFC":"Truist","TFX":"Teleflex","TGT":"Target","TJX":"TJX","TMO":"Thermo Fisher","TMUS":"T-Mobile","TPR":"Tapestry","TRGP":"Targa","TRMB":"Trimble","TROW":"T. Rowe Price","TRV":"Travelers","TSCO":"Tractor Supply","TSLA":"Tesla","TSN":"Tyson","TT":"Trane","TTWO":"Take-Two","TXN":"Texas Instruments","TXT":"Textron","TYL":"Tyler Tech","UAL":"United Airlines","UBER":"Uber","UDR":"UDR","UHS":"Universal Health","ULTA":"Ulta","UNH":"UnitedHealth","UNP":"Union Pacific","UPS":"UPS","URI":"United Rentals","USB":"US Bancorp","V":"Visa","VICI":"VICI","VLO":"Valero","VLTO":"Veralto","VMC":"Vulcan","VRSK":"Verisk","VRSN":"VeriSign","VRTX":"Vertex","VST":"Vistra","VTR":"Ventas","VTRS":"Viatris","VZ":"Verizon","WAB":"Wabtec","WAT":"Waters","WBA":"Walgreens","WBD":"Warner Bros","WDC":"Western Digital","WEC":"WEC Energy","WELL":"Welltower","WFC":"Wells Fargo","WM":"Waste Management","WMB":"Williams","WMT":"Walmart","WRB":"W.R. Berkley","WST":"West Pharma","WTW":"WTW","WY":"Weyerhaeuser","WYNN":"Wynn","XEL":"Xcel","XOM":"Exxon","XYL":"Xylem","YUM":"Yum!","ZBH":"Zimmer Biomet","ZBRA":"Zebra","ZTS":"Zoetis"}

@dataclass
class Stock:
    ticker: str
    name: str
    price: float
    mean: float
    std: float
    z: float
    gap: float
    gap_pct: float
    rsi: float
    half_life: float
    prob: float
    days: float
    signal: str
    direction: str
    prices: List[float]
    dates: List[str]
    gap_hist: List[float]

def fetch_data(ticker, days=120):
    try:
        import yfinance as yf
        df = yf.Ticker(ticker).history(start=(datetime.now()-timedelta(days=days)).strftime('%Y-%m-%d'), end=datetime.now().strftime('%Y-%m-%d'), raise_errors=False)
        if df is None or df.empty or len(df)<50: return None
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            if df['date'].dt.tz: df['date'] = df['date'].dt.tz_localize(None)
        return df[['date','close']].tail(100)
    except: return None

def fetch_all():
    global stock_data_cache, fetch_status
    import time
    fetch_status = {"in_progress":True,"completed":0,"failed":0,"total":len(SP500),"message":"Starting...","last_fetch":None}
    cache = {}
    for i,t in enumerate(SP500):
        fetch_status["message"] = f"Fetching {t}... ({i+1}/{len(SP500)})"
        df = fetch_data(t)
        if df is not None: cache[t]=df; fetch_status["completed"]+=1
        else: fetch_status["failed"]+=1
        if (i+1)%10==0: time.sleep(0.5)
    stock_data_cache = cache
    fetch_status["in_progress"]=False
    fetch_status["last_fetch"]=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    fetch_status["message"]=f"Done: {fetch_status['completed']} loaded"

def rsi(prices,p=14):
    if len(prices)<p+1: return 50.0
    d=np.diff(prices); g,l=np.where(d>0,d,0),np.where(d<0,-d,0)
    a=1.0/p; ag,al=g[0],l[0]
    for i in range(1,len(g)): ag=a*g[i]+(1-a)*ag; al=a*l[i]+(1-a)*al
    if al<0.0001: return 95.0 if ag>0 else 50.0
    return max(5,min(95,100-(100/(1+ag/al))))

def half_life(prices):
    if len(prices)<20: return 30.0
    try:
        y,x=np.diff(prices),prices[:-1]
        b=np.linalg.lstsq(np.column_stack([np.ones(len(x)),x]),y,rcond=None)[0][1]
        if b>=0: return 45.0
        return min(max(-np.log(2)/b,3),60)
    except: return 30.0

def prob(z,r,hl):
    zp=stats.norm.cdf(abs(z))-0.5; zp=0.4+0.4*(2*zp)
    if r<30: rp=0.5+0.4*(30-r)/30
    elif r>70: rp=0.5+0.4*(r-70)/30
    else: rp=0.3+0.2*min(abs(r-50),20)/20
    hp=0.7+0.2*(30-hl)/30 if hl<30 else 0.3+0.4*max(0,(60-hl))/60
    ag=1.15 if (z<0 and r<40) or (z>0 and r>60) else (0.85 if (z<0 and r>60) or (z>0 and r<40) else 1.0)
    return min(max((zp*0.35+rp*0.35+hp*0.30)*ag,0.15),0.92)

def analyze(t,df):
    if df is None or len(df)<50: return None
    p=df['close'].values; d=df['date'].dt.strftime('%Y-%m-%d').tolist()
    cur,m,s=p[-1],np.mean(p),np.std(p)
    if s<0.01: return None
    z=(cur-m)/s; gap=cur-m; gap_pct=(gap/m)*100
    gh=[(x-m) for x in p]
    r,hl=rsi(p),half_life(p)
    pr=prob(z,r,hl)
    days=min(max(hl*(1+0.5*abs(z)),3),45)
    az=abs(z)
    if az>2.0 and ((z<0 and r<35) or (z>0 and r>65)): sig="STRONG"
    elif az>1.8 and (r<40 or r>60): sig="MODERATE"
    elif az>1.5: sig="WEAK"
    else: sig="MINIMAL"
    dir="LONG (Oversold)" if z<0 else "SHORT (Overbought)"
    return Stock(t,NAMES.get(t,t),round(cur,2),round(m,2),round(s,2),round(z,2),round(gap,2),round(gap_pct,2),round(r,1),round(hl,1),round(pr,3),round(days,1),sig,dir,[round(x,2) for x in p],d,[round(x,2) for x in gh])

@app.route('/api/fetch', methods=['POST'])
def api_fetch():
    if fetch_status["in_progress"]: return jsonify({"error":"Already fetching"})
    t=threading.Thread(target=fetch_all); t.daemon=True; t.start()
    return jsonify({"status":"started"})

@app.route('/api/status')
def api_status():
    return jsonify({"stocks_loaded":len(stock_data_cache),"in_progress":fetch_status["in_progress"],"completed":fetch_status["completed"],"failed":fetch_status["failed"],"total":fetch_status["total"],"message":fetch_status["message"],"last_fetch":fetch_status["last_fetch"]})

@app.route('/api/analyze')
def api_analyze():
    if not stock_data_cache: return jsonify({"error":"No data","results":[]})
    results=[]
    for t,df in stock_data_cache.items():
        a=analyze(t,df)
        if a and abs(a.z)>=1.0: results.append(a)
    results.sort(key=lambda x:x.prob,reverse=True)
    top=results[:10]
    return jsonify({"results":[{"ticker":r.ticker,"company_name":r.name,"current_price":r.price,"mean_price":r.mean,"std_dev":r.std,"z_score":r.z,"gap_from_mean":r.gap,"gap_percentage":r.gap_pct,"rsi":r.rsi,"half_life":r.half_life,"reversion_probability":r.prob,"expected_days":r.days,"signal_strength":r.signal,"direction":r.direction,"prices":r.prices,"dates":r.dates,"gap_history":r.gap_hist} for r in top],"total_analyzed":len(stock_data_cache),"candidates_found":len(results)})

@app.route('/')
def index(): return HTML

HTML='''<!DOCTYPE html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"><title>S&P 500 Mean Reversion</title><script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script><style>*{margin:0;padding:0;box-sizing:border-box}body{font-family:-apple-system,sans-serif;background:linear-gradient(135deg,#0f172a,#1e293b,#0f172a);min-height:100vh;color:#e2e8f0}.container{max-width:1400px;margin:0 auto;padding:20px}.header{text-align:center;padding:40px 20px;margin-bottom:30px}.header h1{font-size:2.2rem;font-weight:700;background:linear-gradient(135deg,#60a5fa,#34d399);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:10px}.header p{color:#94a3b8}.controls{background:rgba(30,41,59,0.8);border:1px solid rgba(71,85,105,0.5);border-radius:16px;padding:24px;margin-bottom:30px;display:flex;align-items:center;gap:20px;flex-wrap:wrap}.btn{padding:14px 28px;border-radius:10px;border:none;font-size:15px;font-weight:600;cursor:pointer}.btn-primary{background:linear-gradient(135deg,#3b82f6,#2563eb);color:white}.btn-success{background:linear-gradient(135deg,#22c55e,#16a34a);color:white}.btn:disabled{background:#475569;cursor:not-allowed}.btn:hover:not(:disabled){transform:translateY(-2px);box-shadow:0 8px 20px rgba(59,130,246,0.4)}.progress-section{flex:1;min-width:200px}.progress-bar{height:8px;background:#334155;border-radius:4px;overflow:hidden;margin-bottom:8px}.progress-fill{height:100%;background:linear-gradient(90deg,#3b82f6,#22c55e);transition:width 0.3s}.progress-text{font-size:13px;color:#94a3b8}.hidden{display:none!important}.stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin-bottom:30px}.stat-card{background:rgba(30,41,59,0.6);border:1px solid rgba(71,85,105,0.3);border-radius:12px;padding:20px;text-align:center}.stat-label{color:#94a3b8;font-size:12px;text-transform:uppercase}.stat-value{font-size:28px;font-weight:700;margin-top:8px}.stat-value.green{color:#34d399}.stat-value.blue{color:#60a5fa}.results-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:24px}.stock-card{background:rgba(30,41,59,0.8);border:1px solid rgba(71,85,105,0.5);border-radius:16px;padding:24px;transition:all 0.3s}.stock-card:hover{transform:translateY(-4px);box-shadow:0 12px 40px rgba(0,0,0,0.3)}.stock-header{display:flex;justify-content:space-between;margin-bottom:16px;padding-bottom:16px;border-bottom:1px solid rgba(71,85,105,0.3)}.stock-ticker{font-size:22px;font-weight:700}.stock-company{font-size:13px;color:#94a3b8;margin-top:4px}.prob-value{font-size:26px;font-weight:700;color:#34d399}.prob-label{font-size:11px;color:#94a3b8}.metrics-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:16px}.metric{background:rgba(15,23,42,0.5);border-radius:8px;padding:10px;text-align:center}.metric-label{font-size:10px;color:#64748b;text-transform:uppercase}.metric-value{font-size:14px;font-weight:600;margin-top:4px}.metric-value.positive{color:#34d399}.metric-value.negative{color:#f87171}.direction-badge{display:inline-block;padding:6px 14px;border-radius:20px;font-size:11px;font-weight:600;margin-bottom:12px}.direction-badge.long{background:rgba(34,197,94,0.2);color:#34d399}.direction-badge.short{background:rgba(248,113,113,0.2);color:#f87171}.chart-container{height:180px;margin-top:12px;background:rgba(15,23,42,0.3);border-radius:8px;padding:10px}.empty-state{text-align:center;padding:80px 20px;color:#64748b}.empty-state h3{font-size:22px;color:#94a3b8;margin-bottom:12px}.footer{text-align:center;padding:40px 20px;color:#64748b;font-size:13px}</style></head><body><div class="container"><header class="header"><h1>S&P 500 Mean Reversion Analysis</h1><p>Top 10 stocks most likely to revert to their mean within 30 days</p></header><div class="controls"><button class="btn btn-success" id="fetchBtn">Fetch S&P 500 Data</button><div class="progress-section hidden" id="progressSection"><div class="progress-bar"><div class="progress-fill" id="progressFill" style="width:0%"></div></div><p class="progress-text" id="progressText">Starting...</p></div><button class="btn btn-primary" id="analyzeBtn" disabled>Find Top 10 Candidates</button></div><div class="stats-grid"><div class="stat-card"><p class="stat-label">Stocks Loaded</p><p class="stat-value" id="stocksLoaded">0</p></div><div class="stat-card"><p class="stat-label">Candidates Found</p><p class="stat-value green" id="candidatesFound">-</p></div><div class="stat-card"><p class="stat-label">Avg Probability</p><p class="stat-value blue" id="avgProbability">-</p></div><div class="stat-card"><p class="stat-label">Last Updated</p><p class="stat-value" id="lastUpdated" style="font-size:16px">-</p></div></div><div id="resultsContainer"><div class="empty-state"><h3>No Analysis Yet</h3><p>Click "Fetch S&P 500 Data" then "Find Top 10 Candidates"</p></div></div><footer class="footer"><p>Mean Reversion Analysis | Data from Yahoo Finance | For informational purposes only</p></footer></div><script>let charts={},poll=null;document.getElementById("fetchBtn").onclick=function(){this.disabled=true;document.getElementById("progressSection").classList.remove("hidden");document.getElementById("progressFill").style.width="0%";fetch("/api/fetch",{method:"POST"}).then(r=>r.json()).then(()=>{poll=setInterval(pollStatus,1000)})};function pollStatus(){fetch("/api/status").then(r=>r.json()).then(d=>{document.getElementById("stocksLoaded").textContent=d.stocks_loaded;if(d.in_progress){const p=d.total>0?(d.completed/d.total*100):0;document.getElementById("progressFill").style.width=p+"%";document.getElementById("progressText").textContent=d.message}else{clearInterval(poll);document.getElementById("progressFill").style.width="100%";document.getElementById("progressText").textContent=d.message;document.getElementById("fetchBtn").disabled=false;if(d.stocks_loaded>0){document.getElementById("analyzeBtn").disabled=false;if(d.last_fetch)document.getElementById("lastUpdated").textContent=d.last_fetch.split(" ")[1]}}})}fetch("/api/status").then(r=>r.json()).then(d=>{document.getElementById("stocksLoaded").textContent=d.stocks_loaded;if(d.stocks_loaded>0){document.getElementById("analyzeBtn").disabled=false;if(d.last_fetch)document.getElementById("lastUpdated").textContent=d.last_fetch.split(" ")[1]}});document.getElementById("analyzeBtn").onclick=function(){this.disabled=true;this.textContent="Analyzing...";fetch("/api/analyze").then(r=>r.json()).then(d=>{if(d.error)alert(d.error);else render(d);this.disabled=false;this.textContent="Find Top 10 Candidates"})};function render(d){document.getElementById("candidatesFound").textContent=d.candidates_found;if(d.results.length>0){const avg=d.results.reduce((s,r)=>s+r.reversion_probability,0)/d.results.length;document.getElementById("avgProbability").textContent=(avg*100).toFixed(1)+"%"}const c=document.getElementById("resultsContainer");if(d.results.length===0){c.innerHTML='<div class="empty-state"><h3>No Candidates Found</h3></div>';return}let h='<div class="results-grid">';d.results.forEach((s,i)=>{const isL=s.z_score<0;h+=`<div class="stock-card"><div class="stock-header"><div><div class="stock-ticker">#${i+1} ${s.ticker}</div><div class="stock-company">${s.company_name}</div></div><div style="text-align:right"><div class="prob-value">${(s.reversion_probability*100).toFixed(1)}%</div><div class="prob-label">Reversion Prob.</div></div></div><span class="direction-badge ${isL?"long":"short"}">${s.direction}</span><div class="metrics-grid"><div class="metric"><div class="metric-label">Current</div><div class="metric-value">$${s.current_price.toFixed(2)}</div></div><div class="metric"><div class="metric-label">Mean</div><div class="metric-value">$${s.mean_price.toFixed(2)}</div></div><div class="metric"><div class="metric-label">Gap</div><div class="metric-value ${s.gap_percentage<0?"negative":"positive"}">${s.gap_percentage>0?"+":""}${s.gap_percentage.toFixed(1)}%</div></div><div class="metric"><div class="metric-label">Z-Score</div><div class="metric-value ${s.z_score<0?"negative":"positive"}">${s.z_score.toFixed(2)}σ</div></div><div class="metric"><div class="metric-label">RSI</div><div class="metric-value">${s.rsi.toFixed(1)}</div></div><div class="metric"><div class="metric-label">Exp. Days</div><div class="metric-value">${s.expected_days.toFixed(0)}</div></div></div><div class="chart-container"><canvas id="chart-${s.ticker}"></canvas></div></div>`});h+="</div>";c.innerHTML=h;d.results.forEach(s=>chart(s))}function chart(s){const cv=document.getElementById("chart-"+s.ticker);if(!cv)return;if(charts[s.ticker])charts[s.ticker].destroy();const lb=s.dates.map(d=>{const dt=new Date(d);return(dt.getMonth()+1)+"/"+dt.getDate()});charts[s.ticker]=new Chart(cv.getContext("2d"),{type:"line",data:{labels:lb,datasets:[{label:"Gap from Mean ($)",data:s.gap_history,borderColor:s.z_score<0?"#f87171":"#34d399",backgroundColor:s.z_score<0?"rgba(248,113,113,0.1)":"rgba(52,211,153,0.1)",borderWidth:2,fill:true,tension:0.3,pointRadius:0},{label:"Mean (0)",data:Array(s.prices.length).fill(0),borderColor:"#60a5fa",borderWidth:2,borderDash:[5,5],fill:false,pointRadius:0}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:true,position:"top",labels:{color:"#94a3b8",boxWidth:10,padding:6,font:{size:9}}}},scales:{x:{grid:{color:"rgba(71,85,105,0.2)"},ticks:{color:"#64748b",maxTicksLimit:5,font:{size:9}}},y:{grid:{color:"rgba(71,85,105,0.2)"},ticks:{color:"#64748b",font:{size:9},callback:v=>"$"+v.toFixed(0)}}}}})}</script></body></html>'''

if __name__=='__main__':
    port=int(os.environ.get('PORT',5000))
    print("\n"+"="*55+"\n   S&P 500 Mean Reversion Analysis\n"+"="*55)
    print(f"\n   Open: http://127.0.0.1:{port}\n\n   Press Ctrl+C to stop\n"+"="*55+"\n")
    app.run(host='0.0.0.0',port=port,debug=False)
