#!/usr/bin/env python3
"""Mean Reversion Web App - Local Version with Full S&P 500"""

import numpy as np
from scipy import special
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Optional, Dict
import logging
import threading

from flask import Flask, jsonify, request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

stock_data_cache: Dict[str, pd.DataFrame] = {}
fetch_status = {"in_progress": False, "last_fetch": None, "total": 0, "completed": 0, "failed": 0, "message": ""}

SP500_TICKERS = [
    "A", "AAPL", "ABBV", "ABNB", "ABT", "ACGL", "ACN", "ADBE", "ADI", "ADM", "ADP", "ADSK", "AEE", "AEP", "AES", "AFL", "AIG", "AIZ", "AJG", "AKAM",
    "ALB", "ALGN", "ALL", "ALLE", "AMAT", "AMCR", "AMD", "AME", "AMGN", "AMP", "AMT", "AMZN", "ANET", "ANSS", "AON", "AOS", "APA", "APD", "APH", "APTV",
    "ARE", "ATO", "AVB", "AVGO", "AVY", "AWK", "AXON", "AXP", "AZO", "BA", "BAC", "BALL", "BAX", "BBWI", "BBY", "BDX", "BEN", "BG", "BIIB", "BIO",
    "BK", "BKNG", "BKR", "BLDR", "BLK", "BMY", "BR", "BRK-B", "BRO", "BSX", "BWA", "BX", "BXP", "C", "CAG", "CAH", "CARR", "CAT", "CB", "CBOE",
    "CBRE", "CCI", "CCL", "CDNS", "CDW", "CE", "CEG", "CF", "CFG", "CHD", "CHRW", "CHTR", "CI", "CINF", "CL", "CLX", "CMA", "CMCSA", "CME", "CMG",
    "CMI", "CMS", "CNC", "CNP", "COF", "COO", "COP", "COR", "COST", "CPAY", "CPB", "CPRT", "CPT", "CRL", "CRM", "CSCO", "CSGP", "CSX", "CTAS", "CTLT",
    "CTRA", "CTSH", "CTVA", "CVS", "CVX", "D", "DAL", "DAY", "DD", "DE", "DECK", "DFS", "DG", "DGX", "DHI", "DHR", "DIS", "DLR", "DLTR", "DOC",
    "DOV", "DOW", "DPZ", "DRI", "DTE", "DUK", "DVA", "DVN", "DXCM", "EA", "EBAY", "ECL", "ED", "EFX", "EG", "EIX", "EL", "ELV", "EMN", "EMR",
    "ENPH", "EOG", "EPAM", "EQIX", "EQR", "EQT", "ES", "ESS", "ETN", "ETR", "ETSY", "EVRG", "EW", "EXC", "EXPD", "EXPE", "EXR", "F", "FANG", "FAST",
    "FCX", "FDS", "FDX", "FE", "FFIV", "FI", "FICO", "FIS", "FITB", "FLT", "FMC", "FOX", "FOXA", "FRT", "FSLR", "FTNT", "FTV", "GD", "GDDY", "GE",
    "GEHC", "GEN", "GEV", "GILD", "GIS", "GL", "GLW", "GM", "GNRC", "GOOG", "GOOGL", "GPC", "GPN", "GRMN", "GS", "GWW", "HAL", "HAS", "HBAN", "HCA",
    "HD", "HES", "HIG", "HII", "HLT", "HOLX", "HON", "HPE", "HPQ", "HRL", "HSIC", "HST", "HSY", "HUBB", "HUM", "HWM", "IBM", "ICE", "IDXX", "IEX",
    "IFF", "ILMN", "INCY", "INTC", "INTU", "INVH", "IP", "IPG", "IQV", "IR", "IRM", "ISRG", "IT", "ITW", "J", "JBHT", "JBL", "JCI", "JKHY", "JNJ",
    "JNPR", "JPM", "K", "KDP", "KEY", "KEYS", "KHC", "KIM", "KKR", "KLAC", "KMB", "KMI", "KMX", "KO", "KR", "KVUE", "L", "LDOS", "LEN", "LH",
    "LHX", "LIN", "LKQ", "LLY", "LMT", "LNT", "LOW", "LRCX", "LULU", "LUV", "LVS", "LW", "LYB", "LYV", "MA", "MAA", "MAR", "MAS", "MCD", "MCHP",
    "MCK", "MCO", "MDLZ", "MDT", "MET", "META", "MGM", "MHK", "MKC", "MKTX", "MLM", "MMC", "MMM", "MNST", "MO", "MOH", "MOS", "MPC", "MPWR", "MRK",
    "MRNA", "MRO", "MS", "MSCI", "MSFT", "MSI", "MTB", "MTCH", "MTD", "MU", "NCLH", "NDAQ", "NDSN", "NEE", "NEM", "NFLX", "NI", "NKE", "NOC", "NOW",
    "NRG", "NSC", "NTAP", "NTRS", "NUE", "NVDA", "NVR", "NWS", "NWSA", "O", "ODFL", "OKE", "OMC", "ON", "ORCL", "ORLY", "OTIS", "OXY", "PANW", "PARA",
    "PAYC", "PAYX", "PCAR", "PCG", "PEG", "PEP", "PFE", "PFG", "PG", "PGR", "PH", "PHM", "PKG", "PLD", "PM", "PNC", "PNR", "PNW", "PODD", "POOL",
    "PPG", "PPL", "PRU", "PSA", "PSX", "PTC", "PWR", "PXD", "QCOM", "QRVO", "RCL", "REG", "REGN", "RF", "RJF", "RL", "RMD", "ROK", "ROL", "ROP",
    "ROST", "RSG", "RTX", "SBAC", "SBUX", "SCHW", "SHW", "SJM", "SLB", "SMCI", "SNA", "SNPS", "SO", "SOLV", "SPG", "SPGI", "SRE", "STE", "STLD", "STT",
    "STX", "STZ", "SWK", "SWKS", "SYF", "SYK", "SYY", "T", "TAP", "TDG", "TDY", "TECH", "TEL", "TER", "TFC", "TFX", "TGT", "TJX", "TMO", "TMUS",
    "TPR", "TRGP", "TRMB", "TROW", "TRV", "TSCO", "TSLA", "TSN", "TT", "TTWO", "TXN", "TXT", "TYL", "UAL", "UBER", "UDR", "UHS", "ULTA", "UNH",
    "UNP", "UPS", "URI", "USB", "V", "VICI", "VLO", "VLTO", "VMC", "VRSK", "VRSN", "VRTX", "VST", "VTR", "VTRS", "VZ", "WAB", "WAT", "WBA", "WBD",
    "WDC", "WEC", "WELL", "WFC", "WM", "WMB", "WMT", "WRB", "WST", "WTW", "WY", "WYNN", "XEL", "XOM", "XYL", "YUM", "ZBH", "ZBRA", "ZTS"
]

# Company names mapping
COMPANY_NAMES = {
    "A": "Agilent Technologies", "AAPL": "Apple Inc.", "ABBV": "AbbVie Inc.", "ABNB": "Airbnb", "ABT": "Abbott Laboratories",
    "ACGL": "Arch Capital Group", "ACN": "Accenture", "ADBE": "Adobe Inc.", "ADI": "Analog Devices", "ADM": "Archer-Daniels-Midland",
    "ADP": "ADP", "ADSK": "Autodesk", "AEE": "Ameren Corp", "AEP": "American Electric Power", "AES": "AES Corp",
    "AFL": "Aflac", "AIG": "American International Group", "AIZ": "Assurant", "AJG": "Arthur J. Gallagher", "AKAM": "Akamai Technologies",
    "ALB": "Albemarle Corp", "ALGN": "Align Technology", "ALL": "Allstate Corp", "ALLE": "Allegion", "AMAT": "Applied Materials",
    "AMCR": "Amcor", "AMD": "Advanced Micro Devices", "AME": "AMETEK", "AMGN": "Amgen", "AMP": "Ameriprise Financial",
    "AMT": "American Tower", "AMZN": "Amazon", "ANET": "Arista Networks", "ANSS": "ANSYS", "AON": "Aon plc",
    "AOS": "A. O. Smith", "APA": "APA Corp", "APD": "Air Products & Chemicals", "APH": "Amphenol", "APTV": "Aptiv",
    "ARE": "Alexandria Real Estate", "ATO": "Atmos Energy", "AVB": "AvalonBay Communities", "AVGO": "Broadcom Inc.", "AVY": "Avery Dennison",
    "AWK": "American Water Works", "AXON": "Axon Enterprise", "AXP": "American Express", "AZO": "AutoZone", "BA": "Boeing",
    "BAC": "Bank of America", "BALL": "Ball Corp", "BAX": "Baxter International", "BBWI": "Bath & Body Works", "BBY": "Best Buy",
    "BDX": "Becton Dickinson", "BEN": "Franklin Resources", "BG": "Bunge Ltd", "BIIB": "Biogen", "BIO": "Bio-Rad Laboratories",
    "BK": "Bank of New York Mellon", "BKNG": "Booking Holdings", "BKR": "Baker Hughes", "BLDR": "Builders FirstSource", "BLK": "BlackRock",
    "BMY": "Bristol-Myers Squibb", "BR": "Broadridge Financial", "BRK-B": "Berkshire Hathaway", "BRO": "Brown & Brown", "BSX": "Boston Scientific",
    "BWA": "BorgWarner", "BX": "Blackstone", "BXP": "Boston Properties", "C": "Citigroup", "CAG": "Conagra Brands",
    "CAH": "Cardinal Health", "CARR": "Carrier Global", "CAT": "Caterpillar", "CB": "Chubb Ltd", "CBOE": "Cboe Global Markets",
    "CBRE": "CBRE Group", "CCI": "Crown Castle", "CCL": "Carnival Corp", "CDNS": "Cadence Design Systems", "CDW": "CDW Corp",
    "CE": "Celanese Corp", "CEG": "Constellation Energy", "CF": "CF Industries", "CFG": "Citizens Financial Group", "CHD": "Church & Dwight",
    "CHRW": "C.H. Robinson", "CHTR": "Charter Communications", "CI": "Cigna Group", "CINF": "Cincinnati Financial", "CL": "Colgate-Palmolive",
    "CLX": "Clorox", "CMA": "Comerica", "CMCSA": "Comcast", "CME": "CME Group", "CMG": "Chipotle Mexican Grill",
    "CMI": "Cummins", "CMS": "CMS Energy", "CNC": "Centene Corp", "CNP": "CenterPoint Energy", "COF": "Capital One",
    "COO": "Cooper Companies", "COP": "ConocoPhillips", "COR": "Cencora", "COST": "Costco", "CPAY": "Corpay",
    "CPB": "Campbell Soup", "CPRT": "Copart", "CPT": "Camden Property Trust", "CRL": "Charles River Laboratories", "CRM": "Salesforce",
    "CSCO": "Cisco Systems", "CSGP": "CoStar Group", "CSX": "CSX Corp", "CTAS": "Cintas", "CTLT": "Catalent",
    "CTRA": "Coterra Energy", "CTSH": "Cognizant", "CTVA": "Corteva", "CVS": "CVS Health", "CVX": "Chevron",
    "D": "Dominion Energy", "DAL": "Delta Air Lines", "DAY": "Dayforce", "DD": "DuPont", "DE": "Deere & Co",
    "DECK": "Deckers Outdoor", "DFS": "Discover Financial", "DG": "Dollar General", "DGX": "Quest Diagnostics", "DHI": "D.R. Horton",
    "DHR": "Danaher", "DIS": "Walt Disney", "DLR": "Digital Realty", "DLTR": "Dollar Tree", "DOC": "Healthpeak Properties",
    "DOV": "Dover Corp", "DOW": "Dow Inc.", "DPZ": "Domino's Pizza", "DRI": "Darden Restaurants", "DTE": "DTE Energy",
    "DUK": "Duke Energy", "DVA": "DaVita", "DVN": "Devon Energy", "DXCM": "DexCom", "EA": "Electronic Arts",
    "EBAY": "eBay", "ECL": "Ecolab", "ED": "Consolidated Edison", "EFX": "Equifax", "EG": "Everest Group",
    "EIX": "Edison International", "EL": "Estée Lauder", "ELV": "Elevance Health", "EMN": "Eastman Chemical", "EMR": "Emerson Electric",
    "ENPH": "Enphase Energy", "EOG": "EOG Resources", "EPAM": "EPAM Systems", "EQIX": "Equinix", "EQR": "Equity Residential",
    "EQT": "EQT Corp", "ES": "Eversource Energy", "ESS": "Essex Property Trust", "ETN": "Eaton Corp", "ETR": "Entergy",
    "ETSY": "Etsy", "EVRG": "Evergy", "EW": "Edwards Lifesciences", "EXC": "Exelon", "EXPD": "Expeditors International",
    "EXPE": "Expedia Group", "EXR": "Extra Space Storage", "F": "Ford Motor", "FANG": "Diamondback Energy", "FAST": "Fastenal",
    "FCX": "Freeport-McMoRan", "FDS": "FactSet", "FDX": "FedEx", "FE": "FirstEnergy", "FFIV": "F5 Inc",
    "FI": "Fiserv", "FICO": "Fair Isaac Corp", "FIS": "Fidelity National Info", "FITB": "Fifth Third Bancorp", "FLT": "Fleetcor Technologies",
    "FMC": "FMC Corp", "FOX": "Fox Corp Class B", "FOXA": "Fox Corp Class A", "FRT": "Federal Realty", "FSLR": "First Solar",
    "FTNT": "Fortinet", "FTV": "Fortive", "GD": "General Dynamics", "GDDY": "GoDaddy", "GE": "GE Aerospace",
    "GEHC": "GE HealthCare", "GEN": "Gen Digital", "GEV": "GE Vernova", "GILD": "Gilead Sciences", "GIS": "General Mills",
    "GL": "Globe Life", "GLW": "Corning Inc", "GM": "General Motors", "GNRC": "Generac Holdings", "GOOG": "Alphabet Class C",
    "GOOGL": "Alphabet Class A", "GPC": "Genuine Parts", "GPN": "Global Payments", "GRMN": "Garmin", "GS": "Goldman Sachs",
    "GWW": "W.W. Grainger", "HAL": "Halliburton", "HAS": "Hasbro", "HBAN": "Huntington Bancshares", "HCA": "HCA Healthcare",
    "HD": "Home Depot", "HES": "Hess Corp", "HIG": "Hartford Financial", "HII": "Huntington Ingalls", "HLT": "Hilton Worldwide",
    "HOLX": "Hologic", "HON": "Honeywell", "HPE": "Hewlett Packard Enterprise", "HPQ": "HP Inc", "HRL": "Hormel Foods",
    "HSIC": "Henry Schein", "HST": "Host Hotels & Resorts", "HSY": "Hershey", "HUBB": "Hubbell Inc", "HUM": "Humana",
    "HWM": "Howmet Aerospace", "IBM": "IBM", "ICE": "Intercontinental Exchange", "IDXX": "IDEXX Laboratories", "IEX": "IDEX Corp",
    "IFF": "International Flavors", "ILMN": "Illumina", "INCY": "Incyte", "INTC": "Intel", "INTU": "Intuit",
    "INVH": "Invitation Homes", "IP": "International Paper", "IPG": "Interpublic Group", "IQV": "IQVIA Holdings", "IR": "Ingersoll Rand",
    "IRM": "Iron Mountain", "ISRG": "Intuitive Surgical", "IT": "Gartner", "ITW": "Illinois Tool Works", "J": "Jacobs Solutions",
    "JBHT": "J.B. Hunt Transport", "JBL": "Jabil", "JCI": "Johnson Controls", "JKHY": "Jack Henry & Associates", "JNJ": "Johnson & Johnson",
    "JNPR": "Juniper Networks", "JPM": "JPMorgan Chase", "K": "Kellanova", "KDP": "Keurig Dr Pepper", "KEY": "KeyCorp",
    "KEYS": "Keysight Technologies", "KHC": "Kraft Heinz", "KIM": "Kimco Realty", "KKR": "KKR & Co", "KLAC": "KLA Corp",
    "KMB": "Kimberly-Clark", "KMI": "Kinder Morgan", "KMX": "CarMax", "KO": "Coca-Cola", "KR": "Kroger",
    "KVUE": "Kenvue", "L": "Loews Corp", "LDOS": "Leidos Holdings", "LEN": "Lennar", "LH": "Labcorp Holdings",
    "LHX": "L3Harris Technologies", "LIN": "Linde plc", "LKQ": "LKQ Corp", "LLY": "Eli Lilly", "LMT": "Lockheed Martin",
    "LNT": "Alliant Energy", "LOW": "Lowe's", "LRCX": "Lam Research", "LULU": "Lululemon Athletica", "LUV": "Southwest Airlines",
    "LVS": "Las Vegas Sands", "LW": "Lamb Weston", "LYB": "LyondellBasell", "LYV": "Live Nation Entertainment", "MA": "Mastercard",
    "MAA": "Mid-America Apartment", "MAR": "Marriott International", "MAS": "Masco Corp", "MCD": "McDonald's", "MCHP": "Microchip Technology",
    "MCK": "McKesson", "MCO": "Moody's Corp", "MDLZ": "Mondelez International", "MDT": "Medtronic", "MET": "MetLife",
    "META": "Meta Platforms", "MGM": "MGM Resorts", "MHK": "Mohawk Industries", "MKC": "McCormick & Co", "MKTX": "MarketAxess",
    "MLM": "Martin Marietta", "MMC": "Marsh McLennan", "MMM": "3M", "MNST": "Monster Beverage", "MO": "Altria Group",
    "MOH": "Molina Healthcare", "MOS": "Mosaic Co", "MPC": "Marathon Petroleum", "MPWR": "Monolithic Power Systems", "MRK": "Merck & Co",
    "MRNA": "Moderna", "MRO": "Marathon Oil", "MS": "Morgan Stanley", "MSCI": "MSCI Inc", "MSFT": "Microsoft",
    "MSI": "Motorola Solutions", "MTB": "M&T Bank", "MTCH": "Match Group", "MTD": "Mettler-Toledo", "MU": "Micron Technology",
    "NCLH": "Norwegian Cruise Line", "NDAQ": "Nasdaq Inc", "NDSN": "Nordson Corp", "NEE": "NextEra Energy", "NEM": "Newmont",
    "NFLX": "Netflix", "NI": "NiSource", "NKE": "Nike", "NOC": "Northrop Grumman", "NOW": "ServiceNow",
    "NRG": "NRG Energy", "NSC": "Norfolk Southern", "NTAP": "NetApp", "NTRS": "Northern Trust", "NUE": "Nucor",
    "NVDA": "NVIDIA", "NVR": "NVR Inc", "NWS": "News Corp Class B", "NWSA": "News Corp Class A", "O": "Realty Income",
    "ODFL": "Old Dominion Freight", "OKE": "ONEOK", "OMC": "Omnicom Group", "ON": "ON Semiconductor", "ORCL": "Oracle",
    "ORLY": "O'Reilly Automotive", "OTIS": "Otis Worldwide", "OXY": "Occidental Petroleum", "PANW": "Palo Alto Networks", "PARA": "Paramount Global",
    "PAYC": "Paycom Software", "PAYX": "Paychex", "PCAR": "PACCAR", "PCG": "PG&E Corp", "PEG": "Public Service Enterprise",
    "PEP": "PepsiCo", "PFE": "Pfizer", "PFG": "Principal Financial", "PG": "Procter & Gamble", "PGR": "Progressive Corp",
    "PH": "Parker-Hannifin", "PHM": "PulteGroup", "PKG": "Packaging Corp of America", "PLD": "Prologis", "PM": "Philip Morris International",
    "PNC": "PNC Financial", "PNR": "Pentair", "PNW": "Pinnacle West Capital", "PODD": "Insulet Corp", "POOL": "Pool Corp",
    "PPG": "PPG Industries", "PPL": "PPL Corp", "PRU": "Prudential Financial", "PSA": "Public Storage", "PSX": "Phillips 66",
    "PTC": "PTC Inc", "PWR": "Quanta Services", "PXD": "Pioneer Natural Resources", "QCOM": "Qualcomm", "QRVO": "Qorvo",
    "RCL": "Royal Caribbean", "REG": "Regency Centers", "REGN": "Regeneron Pharmaceuticals", "RF": "Regions Financial", "RJF": "Raymond James Financial",
    "RL": "Ralph Lauren", "RMD": "ResMed", "ROK": "Rockwell Automation", "ROL": "Rollins Inc", "ROP": "Roper Technologies",
    "ROST": "Ross Stores", "RSG": "Republic Services", "RTX": "RTX Corp", "SBAC": "SBA Communications", "SBUX": "Starbucks",
    "SCHW": "Charles Schwab", "SHW": "Sherwin-Williams", "SJM": "J.M. Smucker", "SLB": "Schlumberger", "SMCI": "Super Micro Computer",
    "SNA": "Snap-on", "SNPS": "Synopsys", "SO": "Southern Company", "SOLV": "Solventum", "SPG": "Simon Property Group",
    "SPGI": "S&P Global", "SRE": "Sempra", "STE": "STERIS", "STLD": "Steel Dynamics", "STT": "State Street",
    "STX": "Seagate Technology", "STZ": "Constellation Brands", "SWK": "Stanley Black & Decker", "SWKS": "Skyworks Solutions", "SYF": "Synchrony Financial",
    "SYK": "Stryker", "SYY": "Sysco", "T": "AT&T", "TAP": "Molson Coors", "TDG": "TransDigm Group",
    "TDY": "Teledyne Technologies", "TECH": "Bio-Techne", "TEL": "TE Connectivity", "TER": "Teradyne", "TFC": "Truist Financial",
    "TFX": "Teleflex", "TGT": "Target", "TJX": "TJX Companies", "TMO": "Thermo Fisher Scientific", "TMUS": "T-Mobile US",
    "TPR": "Tapestry Inc", "TRGP": "Targa Resources", "TRMB": "Trimble Inc", "TROW": "T. Rowe Price", "TRV": "Travelers Companies",
    "TSCO": "Tractor Supply", "TSLA": "Tesla", "TSN": "Tyson Foods", "TT": "Trane Technologies", "TTWO": "Take-Two Interactive",
    "TXN": "Texas Instruments", "TXT": "Textron", "TYL": "Tyler Technologies", "UAL": "United Airlines", "UBER": "Uber Technologies",
    "UDR": "UDR Inc", "UHS": "Universal Health Services", "ULTA": "Ulta Beauty", "UNH": "UnitedHealth Group", "UNP": "Union Pacific",
    "UPS": "United Parcel Service", "URI": "United Rentals", "USB": "U.S. Bancorp", "V": "Visa", "VICI": "VICI Properties",
    "VLO": "Valero Energy", "VLTO": "Veralto", "VMC": "Vulcan Materials", "VRSK": "Verisk Analytics", "VRSN": "VeriSign",
    "VRTX": "Vertex Pharmaceuticals", "VST": "Vistra", "VTR": "Ventas", "VTRS": "Viatris", "VZ": "Verizon",
    "WAB": "Westinghouse Air Brake", "WAT": "Waters Corp", "WBA": "Walgreens Boots Alliance", "WBD": "Warner Bros. Discovery", "WDC": "Western Digital",
    "WEC": "WEC Energy Group", "WELL": "Welltower", "WFC": "Wells Fargo", "WM": "Waste Management", "WMB": "Williams Companies",
    "WMT": "Walmart", "WRB": "W. R. Berkley", "WST": "West Pharmaceutical", "WTW": "Willis Towers Watson", "WY": "Weyerhaeuser",
    "WYNN": "Wynn Resorts", "XEL": "Xcel Energy", "XOM": "Exxon Mobil", "XYL": "Xylem Inc", "YUM": "Yum! Brands",
    "ZBH": "Zimmer Biomet", "ZBRA": "Zebra Technologies", "ZTS": "Zoetis"
}

@dataclass
class StockAnalysis:
    ticker: str
    company_name: str
    current_price: float
    mean_price: float
    std_dev: float
    z_score: float
    rsi: float
    gap_from_mean: float
    gap_percentage: float
    reversion_probability: float
    expected_days_to_revert: float
    half_life: float
    signal_strength: str
    direction: str
    prices: List[float]
    dates: List[str]
    composite_score: float = 0.0

def fetch_single_stock(ticker, days=120):
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        df = stock.history(start=(datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'), end=datetime.now().strftime('%Y-%m-%d'), raise_errors=False)
        if df is None or df.empty or len(df) < 50:
            return None
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            if df['date'].dt.tz is not None:
                df['date'] = df['date'].dt.tz_localize(None)
        return df[['date', 'close']].tail(100)
    except Exception as e:
        logger.debug(f"Error fetching {ticker}: {e}")
        return None

def fetch_all_stocks_thread():
    global stock_data_cache, fetch_status
    import time
    fetch_status["in_progress"] = True
    fetch_status["total"] = len(SP500_TICKERS)
    fetch_status["completed"] = 0
    fetch_status["failed"] = 0
    fetch_status["message"] = "Starting..."
    fetch_status["last_fetch"] = None
    
    new_cache = {}
    for i, ticker in enumerate(SP500_TICKERS):
        fetch_status["message"] = f"Fetching {ticker}... ({i+1}/{len(SP500_TICKERS)})"
        logger.info(fetch_status["message"])
        df = fetch_single_stock(ticker)
        if df is not None:
            new_cache[ticker] = df
            fetch_status["completed"] += 1
        else:
            fetch_status["failed"] += 1
        if (i + 1) % 10 == 0:
            time.sleep(0.5)
    
    stock_data_cache = new_cache
    fetch_status["in_progress"] = False
    fetch_status["last_fetch"] = datetime.now().strftime('%Y-%m-%d %H:%M')
    fetch_status["message"] = f"Done: {fetch_status['completed']} loaded, {fetch_status['failed']} failed"
    logger.info(fetch_status["message"])

def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    alpha = 1.0 / period
    avg_gain, avg_loss = gains[0], losses[0]
    for i in range(1, len(gains)):
        avg_gain = alpha * gains[i] + (1 - alpha) * avg_gain
        avg_loss = alpha * losses[i] + (1 - alpha) * avg_loss
    if avg_loss < 0.0001:
        return 95.0 if avg_gain > 0 else 50.0
    return max(5, min(95, 100 - (100 / (1 + avg_gain / avg_loss))))

def calculate_half_life(prices):
    if len(prices) < 20:
        return 30.0
    try:
        X = np.column_stack([np.ones(len(prices)-1), prices[:-1]])
        result = np.linalg.lstsq(X, np.diff(prices), rcond=None)
        beta = result[0][1]
        if beta >= 0:
            return 45.0
        return min(max(-np.log(2) / beta, 3), 60)
    except:
        return 30.0

def calculate_reversion_probability(z_score, rsi, half_life):
    z_prob = 0.4 + 0.4 * (1 - 2 * (1 - 0.5 * (1 + special.erf(abs(z_score) / np.sqrt(2)))))
    if rsi < 30:
        rsi_prob = 0.5 + 0.4 * (30 - rsi) / 30
    elif rsi > 70:
        rsi_prob = 0.5 + 0.4 * (rsi - 70) / 30
    else:
        rsi_prob = 0.3 + 0.2 * min(abs(rsi - 50), 20) / 20
    hl_prob = 1 - np.exp(-30 * np.log(2) / half_life) if half_life < 60 else 0.3
    agreement = 1.15 if (z_score < 0 and rsi < 40) or (z_score > 0 and rsi > 60) else (0.85 if (z_score < 0 and rsi > 60) or (z_score > 0 and rsi < 40) else 1.0)
    return min(max((z_prob * 0.35 + rsi_prob * 0.35 + hl_prob * 0.30) * agreement, 0.15), 0.92)

def analyze_stock(ticker, df):
    if df is None or len(df) < 50:
        return None
    prices = df['close'].values
    dates = df['date'].dt.strftime('%Y-%m-%d').tolist()
    current_price, mean_price, std_dev = prices[-1], np.mean(prices), np.std(prices)
    if std_dev < 0.01:
        return None
    z_score = (current_price - mean_price) / std_dev
    rsi = calculate_rsi(prices)
    half_life = calculate_half_life(prices)
    gap_percentage = ((current_price - mean_price) / mean_price) * 100
    reversion_probability = calculate_reversion_probability(z_score, rsi, half_life)
    abs_z = abs(z_score)
    if abs_z > 2.0 and ((z_score < 0 and rsi < 35) or (z_score > 0 and rsi > 65)):
        signal_strength = "STRONG"
    elif abs_z > 1.8 and (rsi < 40 or rsi > 60):
        signal_strength = "MODERATE"
    elif abs_z > 1.5:
        signal_strength = "WEAK"
    else:
        signal_strength = "MINIMAL"
    return StockAnalysis(ticker=ticker, company_name=COMPANY_NAMES.get(ticker, ticker), current_price=round(current_price, 2), mean_price=round(mean_price, 2), std_dev=round(std_dev, 2), z_score=round(z_score, 2), rsi=round(rsi, 1), gap_from_mean=round(current_price - mean_price, 2), gap_percentage=round(gap_percentage, 1), reversion_probability=round(reversion_probability, 3), expected_days_to_revert=round(min(max(half_life * (1 + 0.5 * abs_z), 3), 45), 1), half_life=round(half_life, 1), signal_strength=signal_strength, direction="LONG" if z_score < 0 else "SHORT", prices=prices.tolist(), dates=dates)


@app.route('/')
def index():
    return """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mean Reversion - S&P 500</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460); min-height: 100vh; color: #e5e7eb; }
        .container { max-width: 1400px; margin: 0 auto; padding: 24px 16px; }
        .glass-card { background: rgba(255,255,255,0.05); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; }
        .header { padding: 24px; margin-bottom: 24px; }
        .header h1 { font-size: 28px; font-weight: 700; color: #fff; }
        .header-subtitle { color: #9ca3af; }
        .controls { padding: 20px; margin-bottom: 24px; display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-end; }
        .control-group { display: flex; flex-direction: column; gap: 6px; }
        .control-group label { font-size: 12px; color: #9ca3af; text-transform: uppercase; }
        .control-group input { padding: 10px 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.2); background: rgba(255,255,255,0.1); color: #fff; font-size: 14px; width: 100px; }
        .btn { padding: 12px 24px; border-radius: 8px; border: none; cursor: pointer; font-size: 14px; font-weight: 600; }
        .btn-primary { background: #3b82f6; color: #fff; }
        .btn-primary:hover { background: #2563eb; }
        .btn-primary:disabled { background: #4b5563; cursor: not-allowed; }
        .btn-success { background: #22c55e; color: #fff; }
        .btn-success:hover { background: #16a34a; }
        .btn-success:disabled { background: #4b5563; cursor: not-allowed; }
        .progress-container { margin-top: 10px; }
        .progress-bar { width: 300px; height: 10px; background: #374151; border-radius: 5px; overflow: hidden; }
        .progress-fill { height: 100%; background: linear-gradient(90deg, #3b82f6, #22c55e); transition: width 0.3s; }
        .progress-text { font-size: 12px; color: #9ca3af; margin-top: 5px; }
        .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
        .stat-card { padding: 16px; }
        .stat-label { color: #9ca3af; font-size: 14px; margin-bottom: 4px; }
        .stat-value { font-size: 28px; font-weight: 700; color: #fff; }
        .stat-value.green { color: #22c55e; }
        .stat-value.blue { color: #3b82f6; }
        .main-layout { display: grid; gap: 24px; grid-template-columns: 1fr 2fr; }
        .stock-list { display: flex; flex-direction: column; gap: 12px; max-height: 700px; overflow-y: auto; }
        .stock-card { padding: 16px; cursor: pointer; }
        .stock-card:hover { transform: scale(1.02); }
        .stock-card.selected { border-color: #22c55e; box-shadow: 0 0 20px rgba(34,197,94,0.3); }
        .stock-card-header { display: flex; justify-content: space-between; margin-bottom: 12px; }
        .stock-ticker { font-size: 20px; font-weight: 700; color: #fff; }
        .company-name { font-size: 12px; color: #9ca3af; margin: 2px 0 6px 0; max-width: 180px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .detail-company { font-size: 16px; color: #22c55e; margin-bottom: 4px; }
        .direction-badge { font-size: 12px; padding: 4px 10px; border-radius: 9999px; background: rgba(34,197,94,0.2); color: #22c55e; }
        .stock-price-value { font-size: 22px; font-weight: 700; color: #fff; }
        .stock-gap { font-size: 13px; color: #22c55e; }
        .stock-metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; text-align: center; }
        .metric-label { color: #6b7280; font-size: 11px; }
        .metric-value { font-weight: 700; font-size: 14px; color: #22c55e; }
        .metric-value.blue { color: #3b82f6; }
        .detail-panel { padding: 24px; }
        .detail-header { display: flex; justify-content: space-between; margin-bottom: 16px; }
        .detail-title { font-size: 24px; font-weight: 700; color: #fff; }
        .signal-badge { padding: 8px 16px; border-radius: 8px; font-size: 16px; font-weight: 700; background: rgba(34,197,94,0.2); color: #22c55e; }
        .chart-container { height: 320px; margin-bottom: 24px; }
        .metrics-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
        .metric-box { background: rgba(31,41,55,0.5); border-radius: 8px; padding: 12px; }
        .metric-box-label { color: #6b7280; font-size: 12px; margin-bottom: 4px; }
        .metric-box-value { font-size: 18px; font-weight: 700; color: #fff; }
        .empty-state { text-align: center; padding: 60px 20px; color: #9ca3af; }
        .empty-state h3 { font-size: 18px; margin-bottom: 10px; color: #fff; }
        .status-message { padding: 12px 16px; border-radius: 8px; margin-bottom: 16px; display: none; }
        .status-message.show { display: block; }
        .status-message.error { background: rgba(239,68,68,0.2); color: #ef4444; }
        .status-message.success { background: rgba(34,197,94,0.2); color: #22c55e; }
        .divider { border-left: 1px solid rgba(255,255,255,0.1); height: 60px; margin: 0 16px; }
        .hidden { display: none !important; }
    </style>
</head>
<body>
<div class="container">
    <header class="glass-card header">
        <h1>Mean Reversion Analysis</h1>
        <p class="header-subtitle">S&P 500 - LONG (Oversold) Signals</p>
    </header>

    <div class="glass-card controls">
        <div class="control-group">
            <label>Step 1: Load Data</label>
            <button class="btn btn-success" id="fetchBtn" type="button">Fetch Stock Data</button>
            <div class="progress-container hidden" id="progressContainer">
                <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
                <p class="progress-text" id="progressText">Starting...</p>
            </div>
        </div>
        <div class="divider"></div>
        <div class="control-group">
            <label>Min Z-Score</label>
            <input type="number" id="minZscore" value="1.5" step="0.1" min="0.5" max="4">
        </div>
        <div class="control-group">
            <label>Top Results</label>
            <input type="number" id="topN" value="10" min="5" max="50">
        </div>
        <div class="control-group">
            <label>Step 2: Analyze</label>
            <button class="btn btn-primary" id="analyzeBtn" type="button" disabled>Run Analysis</button>
        </div>
    </div>
    
    <div id="statusMessage" class="status-message"></div>

    <div class="stats-grid">
        <div class="glass-card stat-card"><p class="stat-label">Stocks Loaded</p><p class="stat-value" id="statLoaded">0</p></div>
        <div class="glass-card stat-card"><p class="stat-label">Candidates</p><p class="stat-value green" id="statCandidates">-</p></div>
        <div class="glass-card stat-card"><p class="stat-label">Avg Z-Score</p><p class="stat-value green" id="statZscore">-</p></div>
        <div class="glass-card stat-card"><p class="stat-label">Avg Probability</p><p class="stat-value blue" id="statProb">-</p></div>
    </div>

    <div class="main-layout">
        <div class="stock-list" id="stockList">
            <div class="empty-state"><h3>No Results Yet</h3><p>1. Click "Fetch Stock Data"</p><p>2. Click "Run Analysis"</p></div>
        </div>
        <div class="glass-card detail-panel" id="detailPanel">
            <div class="empty-state"><h3>Select a Stock</h3><p>Run analysis first</p></div>
        </div>
    </div>
</div>

<script>
(function() {
    var stockData = [];
    var selectedTicker = null;
    var detailChart = null;
    var pollTimer = null;

    var fetchBtn = document.getElementById('fetchBtn');
    var analyzeBtn = document.getElementById('analyzeBtn');
    var progressContainer = document.getElementById('progressContainer');
    var progressFill = document.getElementById('progressFill');
    var progressText = document.getElementById('progressText');
    var statusMessage = document.getElementById('statusMessage');

    function showStatus(msg, type) {
        statusMessage.textContent = msg;
        statusMessage.className = 'status-message show ' + type;
    }

    function hideStatus() {
        statusMessage.className = 'status-message';
    }

    function startFetch() {
        console.log('Fetch button clicked!');
        fetchBtn.disabled = true;
        progressContainer.classList.remove('hidden');
        progressFill.style.width = '0%';
        progressText.textContent = 'Starting fetch...';
        hideStatus();

        fetch('/api/fetch', { method: 'POST' })
            .then(function(response) { return response.json(); })
            .then(function(data) {
                console.log('Fetch started:', data);
                if (data.error) {
                    showStatus(data.error, 'error');
                    fetchBtn.disabled = false;
                    progressContainer.classList.add('hidden');
                } else {
                    pollTimer = setInterval(pollStatus, 1000);
                }
            })
            .catch(function(err) {
                console.error('Fetch error:', err);
                showStatus('Error: ' + err.message, 'error');
                fetchBtn.disabled = false;
                progressContainer.classList.add('hidden');
            });
    }

    function pollStatus() {
        fetch('/api/status')
            .then(function(response) { return response.json(); })
            .then(function(data) {
                console.log('Status:', data);
                document.getElementById('statLoaded').textContent = data.stocks_loaded;

                if (data.fetch_in_progress) {
                    var pct = data.fetch_total > 0 ? (data.fetch_completed / data.fetch_total * 100) : 0;
                    progressFill.style.width = pct + '%';
                    progressText.textContent = data.fetch_message || 'Fetching...';
                } else {
                    clearInterval(pollTimer);
                    pollTimer = null;
                    progressFill.style.width = '100%';
                    progressText.textContent = data.fetch_message || 'Complete!';
                    
                    if (data.stocks_loaded > 0) {
                        showStatus('Loaded ' + data.stocks_loaded + ' stocks', 'success');
                        analyzeBtn.disabled = false;
                    }
                    fetchBtn.disabled = false;
                }
            })
            .catch(function(err) {
                console.error('Poll error:', err);
            });
    }

    function runAnalysis() {
        var minZ = parseFloat(document.getElementById('minZscore').value) || 1.5;
        var topN = parseInt(document.getElementById('topN').value) || 10;

        analyzeBtn.disabled = true;
        analyzeBtn.textContent = 'Analyzing...';
        hideStatus();

        fetch('/api/analyze?min_z_score=' + minZ + '&top_n=' + topN)
            .then(function(response) { return response.json(); })
            .then(function(data) {
                console.log('Analysis:', data);
                if (data.error) {
                    showStatus(data.error, 'error');
                } else {
                    stockData = data.results;
                    selectedTicker = stockData.length > 0 ? stockData[0].ticker : null;
                    renderStats();
                    renderStockList();
                    renderDetailPanel();
                    showStatus('Found ' + stockData.length + ' candidates', 'success');
                }
                analyzeBtn.disabled = false;
                analyzeBtn.textContent = 'Run Analysis';
            })
            .catch(function(err) {
                showStatus('Error: ' + err.message, 'error');
                analyzeBtn.disabled = false;
                analyzeBtn.textContent = 'Run Analysis';
            });
    }

    function renderStats() {
        if (!stockData.length) return;
        document.getElementById('statCandidates').textContent = stockData.length;
        var avgZ = stockData.reduce(function(s, x) { return s + x.z_score; }, 0) / stockData.length;
        var avgP = stockData.reduce(function(s, x) { return s + x.reversion_probability; }, 0) / stockData.length * 100;
        document.getElementById('statZscore').textContent = avgZ.toFixed(2);
        document.getElementById('statProb').textContent = avgP.toFixed(1) + '%';
    }

    function renderStockList() {
        var container = document.getElementById('stockList');
        if (!stockData.length) {
            container.innerHTML = '<div class="empty-state"><h3>No Candidates</h3></div>';
            return;
        }
        var html = '';
        for (var i = 0; i < stockData.length; i++) {
            var s = stockData[i];
            html += '<div class="glass-card stock-card' + (s.ticker === selectedTicker ? ' selected' : '') + '" data-ticker="' + s.ticker + '">';
            html += '<div class="stock-card-header"><div><div class="stock-ticker">' + s.ticker + '</div><div class="company-name">' + s.company_name + '</div><span class="direction-badge">LONG</span></div>';
            html += '<div><div class="stock-price-value">$' + s.current_price.toFixed(2) + '</div>';
            html += '<div class="stock-gap">' + s.gap_percentage.toFixed(1) + '% below mean</div></div></div>';
            html += '<div class="stock-metrics"><div><p class="metric-label">Z-Score</p><p class="metric-value">' + s.z_score.toFixed(2) + '</p></div>';
            html += '<div><p class="metric-label">RSI</p><p class="metric-value">' + s.rsi.toFixed(1) + '</p></div>';
            html += '<div><p class="metric-label">Prob</p><p class="metric-value blue">' + (s.reversion_probability * 100).toFixed(0) + '%</p></div></div></div>';
        }
        container.innerHTML = html;

        var cards = container.querySelectorAll('.stock-card');
        for (var j = 0; j < cards.length; j++) {
            cards[j].addEventListener('click', function() {
                selectedTicker = this.getAttribute('data-ticker');
                renderStockList();
                renderDetailPanel();
            });
        }
    }

    function renderDetailPanel() {
        var panel = document.getElementById('detailPanel');
        var s = null;
        for (var i = 0; i < stockData.length; i++) {
            if (stockData[i].ticker === selectedTicker) { s = stockData[i]; break; }
        }
        if (!s) {
            panel.innerHTML = '<div class="empty-state"><h3>Select a Stock</h3></div>';
            return;
        }
        panel.innerHTML = '<div class="detail-header"><div><h2 class="detail-title">' + s.ticker + '</h2><p class="detail-company">' + s.company_name + '</p><p style="color:#9ca3af">100-Day History</p></div><div class="signal-badge">LONG</div></div>' +
            '<div class="chart-container"><canvas id="detailChart"></canvas></div>' +
            '<div class="metrics-grid">' +
            '<div class="metric-box"><p class="metric-box-label">Price</p><p class="metric-box-value">$' + s.current_price.toFixed(2) + '</p></div>' +
            '<div class="metric-box"><p class="metric-box-label">Mean</p><p class="metric-box-value">$' + s.mean_price.toFixed(2) + '</p></div>' +
            '<div class="metric-box"><p class="metric-box-label">Std Dev</p><p class="metric-box-value">$' + s.std_dev.toFixed(2) + '</p></div>' +
            '<div class="metric-box"><p class="metric-box-label">Gap</p><p class="metric-box-value" style="color:#22c55e">' + s.gap_percentage.toFixed(1) + '%</p></div>' +
            '<div class="metric-box"><p class="metric-box-label">Z-Score</p><p class="metric-box-value" style="color:#22c55e">' + s.z_score.toFixed(2) + '</p></div>' +
            '<div class="metric-box"><p class="metric-box-label">RSI</p><p class="metric-box-value" style="color:#22c55e">' + s.rsi.toFixed(1) + '</p></div>' +
            '<div class="metric-box"><p class="metric-box-label">Half-Life</p><p class="metric-box-value">' + s.half_life.toFixed(1) + 'd</p></div>' +
            '<div class="metric-box"><p class="metric-box-label">Prob</p><p class="metric-box-value" style="color:#3b82f6">' + (s.reversion_probability * 100).toFixed(1) + '%</p></div></div>';
        
        if (s.prices && s.prices.length > 0) renderChart(s);
    }

    function renderChart(s) {
        var canvas = document.getElementById('detailChart');
        if (!canvas) return;
        if (detailChart) detailChart.destroy();
        
        var m = s.mean_price, st = s.std_dev;
        var labels = [];
        for (var i = 0; i < s.dates.length; i++) {
            var dt = new Date(s.dates[i]);
            labels.push((dt.getMonth()+1) + '/' + dt.getDate());
        }
        var meanArr = [], p1 = [], m1 = [], p2 = [], m2 = [];
        for (var j = 0; j < s.prices.length; j++) {
            meanArr.push(m); p1.push(m+st); m1.push(m-st); p2.push(m+2*st); m2.push(m-2*st);
        }
        
        detailChart = new Chart(canvas.getContext('2d'), {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    { label: 'Price', data: s.prices, borderColor: '#3b82f6', borderWidth: 2, fill: false, tension: 0.1, pointRadius: 0 },
                    { label: 'Mean', data: meanArr, borderColor: '#22c55e', borderWidth: 2, fill: false, pointRadius: 0 },
                    { label: '+1s', data: p1, borderColor: '#f59e0b', borderWidth: 1, borderDash: [5,5], fill: false, pointRadius: 0 },
                    { label: '-1s', data: m1, borderColor: '#f59e0b', borderWidth: 1, borderDash: [5,5], fill: false, pointRadius: 0 },
                    { label: '+2s', data: p2, borderColor: '#ef4444', borderWidth: 1, borderDash: [3,3], fill: false, pointRadius: 0 },
                    { label: '-2s', data: m2, borderColor: '#ef4444', borderWidth: 1, borderDash: [3,3], fill: false, pointRadius: 0 }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { position: 'top', labels: { color: '#9ca3af' } } },
                scales: {
                    x: { grid: { color: 'rgba(55,65,81,0.5)' }, ticks: { color: '#9ca3af', maxTicksLimit: 10 } },
                    y: { grid: { color: 'rgba(55,65,81,0.5)' }, ticks: { color: '#9ca3af' } }
                }
            }
        });
    }

    // Attach event listeners
    fetchBtn.onclick = startFetch;
    analyzeBtn.onclick = runAnalysis;

    // Check initial status
    fetch('/api/status').then(function(r) { return r.json(); }).then(function(d) {
        document.getElementById('statLoaded').textContent = d.stocks_loaded;
        if (d.stocks_loaded > 0) analyzeBtn.disabled = false;
    }).catch(function() {});

    console.log('App initialized');
})();
</script>
</body>
</html>"""


@app.route('/api/fetch', methods=['POST'])
def api_fetch():
    global fetch_status
    logger.info("API /api/fetch called")
    if fetch_status["in_progress"]:
        return jsonify({"error": "Fetch already in progress"})
    
    thread = threading.Thread(target=fetch_all_stocks_thread)
    thread.daemon = True
    thread.start()
    logger.info("Fetch thread started")
    return jsonify({"status": "started", "total": len(SP500_TICKERS)})


@app.route('/api/status')
def api_status():
    return jsonify({
        "stocks_loaded": len(stock_data_cache),
        "last_fetch": fetch_status["last_fetch"],
        "fetch_in_progress": fetch_status["in_progress"],
        "fetch_total": fetch_status["total"],
        "fetch_completed": fetch_status["completed"],
        "fetch_failed": fetch_status["failed"],
        "fetch_message": fetch_status["message"]
    })


@app.route('/api/analyze')
def api_analyze():
    if len(stock_data_cache) == 0:
        return jsonify({"error": "No data loaded. Click 'Fetch Stock Data' first.", "results": []})
    try:
        min_z = float(request.args.get('min_z_score', 1.5))
        top_n = int(request.args.get('top_n', 10))
        results = []
        for ticker, df in stock_data_cache.items():
            a = analyze_stock(ticker, df)
            if a and a.z_score <= -min_z:
                a.composite_score = a.reversion_probability * (1 + abs(a.z_score) / 3)
                results.append(a)
        results.sort(key=lambda x: x.composite_score, reverse=True)
        return jsonify({
            "results": [{
                "ticker": r.ticker, "company_name": r.company_name, "current_price": r.current_price, "mean_price": r.mean_price,
                "std_dev": r.std_dev, "z_score": r.z_score, "rsi": r.rsi,
                "gap_from_mean": r.gap_from_mean, "gap_percentage": r.gap_percentage,
                "reversion_probability": r.reversion_probability, "expected_days": r.expected_days_to_revert,
                "half_life": r.half_life, "signal_strength": r.signal_strength,
                "direction": r.direction, "prices": r.prices, "dates": r.dates
            } for r in results[:top_n]],
            "count": min(len(results), top_n)
        })
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({"error": str(e), "results": []})


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    print("\n" + "="*50)
    print("Starting Mean Reversion Analysis Server")
    print("="*50)
    print(f"\nOpen your browser to: http://127.0.0.1:{port}")
    print("\nPress Ctrl+C to stop the server\n")
    app.run(host='0.0.0.0', port=port, debug=False)
