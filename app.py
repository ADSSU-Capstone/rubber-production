import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    mean_absolute_percentage_error, r2_score,
)
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")
import io

st.set_page_config(page_title="Rubber Production Dashboard", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
.stApp { background: #f8fafc; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1420px; }
[data-testid="stSidebar"] { background: #f1f5f9; border-right: 1px solid #e2e8f0; }
[data-testid="stSidebar"] * { color: #334155 !important; }
h1, h2, h3, h4 { color: #1e3a8a; font-weight: 700; }
.metric-card { background: #ffffff; border-radius: 12px; padding: 18px 20px; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(15,23,42,0.05); border-top: 3px solid #1e3a8a; }
.metric-label { color: #64748b; font-size: 0.68rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; }
.metric-value { color: #0f172a; font-size: 1.55rem; font-weight: 800; margin-top: 6px; }
.metric-sub { color: #d97706; font-size: 0.75rem; font-weight: 600; margin-top: 4px; }
.note { background: #fffbeb; border-left: 3px solid #d97706; padding: 12px 16px; border-radius: 6px; color: #78350f; font-size: 0.85rem; line-height: 1.55; margin-top: 12px; }
.caveat { background: #fef2f2; border-left: 3px solid #dc2626; padding: 12px 16px; border-radius: 6px; color: #7f1d1d; font-size: 0.85rem; line-height: 1.55; margin-top: 12px; }
.quality { background: #f0f9ff; border-left: 3px solid #0284c7; padding: 12px 16px; border-radius: 6px; color: #0c4a6e; font-size: 0.85rem; line-height: 1.55; margin-top: 12px; }
.method { background: #f8fafc; border-left: 3px solid #64748b; padding: 10px 14px; border-radius: 6px; color: #334155; font-size: 0.82rem; line-height: 1.5; margin-top: 6px; }
.sidebar-title { color: #1e3a8a; margin: 0; font-size: 1.05rem; font-weight: 800; }
.sidebar-sub { color: #64748b; font-size: 0.72rem; margin-top: 2px; }
.stTabs [data-baseweb="tab-list"] { gap: 10px; background: transparent; border-bottom: none; padding: 4px 0; }
.stTabs [data-baseweb="tab"] { background: #ffffff; border: 2px solid #e2e8f0; border-radius: 10px; padding: 10px 20px; color: #475569; font-weight: 600; font-size: 0.88rem; }
.stTabs [data-baseweb="tab"]:hover { border-color: #1e3a8a; color: #1e3a8a; }
.stTabs [aria-selected="true"] { background: #1e3a8a !important; color: #ffffff !important; border-color: #1e3a8a !important; }
.stTabs [data-baseweb="tab-highlight"] { display: none; }
.stTabs [data-baseweb="tab-border"] { display: none; }
</style>
""", unsafe_allow_html=True)

VALID_BARANGAYS = {"San Andres", "Consuelo", "Imelda", "Bun.Brook", "San Teodoro"}
VALID_VARIETIES = {"RRIM 600", "PB 280"}
BARANGAY_ALIASES = {
    "bun.brrook": "Bun.Brook", "bunbrook": "Bun.Brook", "bun.brook": "Bun.Brook",
    "santeodoro": "San Teodoro", "san teodoro": "San Teodoro",
    "sanandres": "San Andres", "san andres": "San Andres",
    "consuelo": "Consuelo", "imelda": "Imelda",
}


def _clean_text(val):
    if pd.isna(val):
        return None
    s = str(val).strip()
    s = " ".join(s.split())
    s = "".join(ch for ch in s if ch.isprintable())
    if s == "" or s.lower() in {"nan", "none", "n/a", "/", "-"}:
        return None
    return s


def _clean_tick(val):
    if pd.isna(val):
        return None
    s = str(val).strip()
    if s == "/" or s.lower() in {"yes", "y", "x"}:
        return "yes"
    return _clean_text(val)


def _clean_number(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)
    s = str(val).replace(",", "").strip()
    if s == "" or s.lower() in {"nan", "none", "/", "-"}:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _normalize_barangay(name):
    if name is None:
        return None
    key = name.strip().lower().replace("  ", " ")
    if key in BARANGAY_ALIASES:
        return BARANGAY_ALIASES[key]
    return BARANGAY_ALIASES.get(key.replace(" ", ""), name.strip())


def _normalize_variety(name):
    if name is None:
        return "Unknown"
    s = " ".join(str(name).strip().split())
    for v in VALID_VARIETIES:
        if s.upper().replace(" ", "") == v.upper().replace(" ", ""):
            return v
    return "Unknown"


EXAMPLE_DATA = """July 2024|Sahay|Jerry||San Andres|RRIM 600|480||2000
July 2024|Sagrado|Improso||San Andres|RRIM 600|35000||5000
July 2024|Roilo|Sheena|N.|San Andres|RRIM 600|480||
July 2024|Sanggo|Jerson||San Andres|RRIM 600|960||
July 2024|Ayangod|Carlina|D.|San Andres|RRIM 600|960||
July 2024|Dacuag|Carmelita||San Andres|RRIM 600|480||
July 2024|Gantalao|Corry|D.|San Andres|RRIM 600|480||
July 2024|Dicang|Dormita|B|San Andres|RRIM 600|720||
July 2024|Rein|Jason||San Andres|RRIM 600|480||200
July 2024|Jimenez|Danilo||San Andres|RRIM 600|1440||500
July 2024|Ayangod|Semon||San Andres|RRIM 600|720||
July 2024|Sabangan|Litto||San Andres|RRIM 600|480||
July 2024|Ocat|Juliet||San Andres|RRIM 600|240||
July 2024|Dela Cruz|Ludivina||San Andres|RRIM 600|240||
July 2024|Ayangod|kim||San Andres|RRIM 600|720||
July 2024|Banno|Domingo||San Andres|RRIM 600|480||100
July 2024|Mangibin|Leaster John||San Andres|RRIM 600|480||
July 2024|Mollaneda|Rona Joy||San Andres|RRIM 600|480||110
July 2024|Cayabas|Martina||San Andres|RRIM 600|480||90
July 2024|Ayangod|Myrna||San Andres|RRIM 600|960||
July 2024|Ayangod|Joeredella|T.|San Andres|RRIM 600|240||
July 2024|Mateo|Noel||San Andres|RRIM 600|720||
July 2024|Mateo|Caridad||San Andres|RRIM 600|720||
July 2024|Banno|Emmanuel||San Andres|RRIM 600|720||110
July 2024|Gula|Efren|M|San Andres|RRIM 600|240||80
July 2024|Magoncia|Jose|R.|San Andres|RRIM 600|240||105
July 2024|Ocat|Ronel|S.|San Andres|RRIM 600|240||70
July 2024|Crodua|charigine|E.|San Andres|RRIM 600|480||100
July 2024|Valdez|Benjamin|Q.|San Andres|RRIM 600|240||90
July 2024|Sentillas|Gilbert|N.|San Andres|RRIM 600|960||300
July 2024|Roilo|Jhon Lerenz|N.|San Andres|RRIM 600|1680||700
July 2024|Roilo|Henry|J.|San Andres|RRIM 600|1440||350
July 2024|Roda|Jerry|C.|San Andres|RRIM 600|480||
July 2024|Roda|Rodito Jr|C.|San Andres|RRIM 600|480||150
July 2024|Roda|Dennis|C.|San Andres|RRIM 600|480||120
July 2024|Escototo|Jesoniel|C.|San Andres|RRIM 600|480||
July 2024|Mangibin|Emily|M.|San Andres|RRIM 600|480||500
July 2024|Claro|Pepsinilla|T.|San Andres|RRIM 600|1440||600
July 2024|Salas|Candelario|H.|San Andres|RRIM 600|1440||350
July 2024|Palagan|Mary Ann|D.|San Andres|RRIM 600|720||260
July 2024|Bisoy|Norjon|D.|San Andres|RRIM 600|480||300
July 2024|Domingo|Danilo|P.|San Andres|RRIM 600|720||200
July 2024|Magoncia|Roberto 111|C.|San Andres|RRIM 600|960||150
July 2024|Walley|Marlon||San Andres|RRIM 600|480||180
July 2024|Galagala|Narciso||Consuelo|PB 280|1920||100
July 2024|Manog|Rolando||Consuelo|RRIM 600|240||
July 2024|Lozada|Melchor||Consuelo|RRIM 600|240||100
July 2024|Buenaflor|Herbert||Consuelo|RRIM 600|1440||400
July 2024|Galagala|Joseph||Consuelo|PB 280|960||350
July 2024|Galagala|Mamerto||Consuelo|PB 280|960||100
July 2024|Duhaylungsod|Sary||Consuelo|RRIM 600|480||900
July 2024|Mique|Arsenio||Consuelo|RRIM 600|480||300
July 2024|Galagala|Emelio||Consuelo|PB 280|1440||230
July 2024|Osic|Ronnie||Consuelo|RRIM 600|720||100
July 2024|Galos|Simplicio||Consuelo|RRIM 600|480||
July 2024|Escatron|Elden||Consuelo|RRIM 600|480||1400
July 2024|Blagantio|Ruth|I.|Consuelo|RRIM 600|6240||180
July 2024|Undag|Malaquias||Consuelo|RRIM 600|720||
July 2024|Casinto|Rolando||Consuelo|RRIM 600|360||
July 2024|Ramos|Alma||Consuelo|RRIM 600|360||80
July 2024|Dawing|Rodrigo||Imelda|RRIM 600|500||150
July 2024|Alejo|Manuel Sr.||Imelda|RRIM 600|250||
July 2024|Dawing|Ruth||Imelda|RRIM 600|500||100
July 2024|Siploc|norma||Imelda|RRIM 600|875||180
July 2024|Alejo|Romulo|H|Imelda|RRIM 600|250||100
July 2024|Dicang|lydia|B|Imelda|RRIM 600|100||80
July 2024|Danquis|Lourdes|B|Imelda|RRIM 600|70||80
July 2024|Salut|Jovita|B|Imelda|RRIM 600|100||
July 2024|Donaire|Elenita||Imelda|RRIM 600|500||
July 2024|Calixtro|Elmer|B|Imelda|RRIM 600|1000||
July 2024|Vellarmino|Esteban,JR.|G.|Imelda|RRIM 600|1000||300
July 2024|Calixtro|Rolando|B|Imelda|RRIM 600|960||
July 2024|Cason|Marlyn|T|Imelda|RRIM 600|480||
July 2024|Mabandos|Melanie|P|Imelda|RRIM 600|960||180
July 2024|Vellarmino|Gerry|G|Imelda|RRIM 600|480||
July 2024|Dumalagan|Judith|P.|Imelda|RRIM 600|720||
July 2024|Allawan|Reymark|C|Imelda|RRIM 600|1440||
July 2024|Donaire|Jennelyn|G|Imelda|RRIM 600|480||
July 2024|Tudtud|Federico|P|Imelda|RRIM 600|480||
July 2024|Calixtro|Charamy|E|Imelda|RRIM 600|480||
July 2024|Fuentes|Mere Irish||Imelda|RRIM 600|480||
July 2024|Tadle|Arlene||Imelda|RRIM 600|480||
July 2024|Lastra|Arante|D.|Imelda|RRIM 600|1920||
July 2024|Equipelag|Haide||Imelda|RRIM 600|240||
July 2024|Corpin|Leonida|V|Imelda|RRIM 600|960||
July 2024|Villarmino|Jerson|B|Imelda|RRIM 600|960||
July 2024|Oroc|Cristita|A|Imelda|RRIM 600|1440||
July 2024|Oroc|Marilyn|C|Imelda|RRIM 600|480||
July 2024|Acebuche|MaryRose|V|Imelda|RRIM 600|960||
July 2024|Oroc|Erico|A|Imelda|RRIM 600|1440||
July 2024|Fuentes|Johna||Imelda|RRIM 600|960||
July 2024|Acebuche|Rowena|V|Imelda|RRIM 600|4800||
July 2024|Villarmino|Jegger||Imelda|RRIM 600|480||
July 2024|Amalio|Reynalen|L|Imelda|RRIM 600|480||
July 2024|Geraldes|Roselyn|T|Imelda|RRIM 600|960||
July 2024|Saldo|Cora||Imelda|RRIM 600|960||
July 2024|Marajay|Amalio||Imelda|RRIM 600|240||
July 2024|Villarmino|Delia||Imelda|RRIM 600|960||
July 2024|Amalio|Richard|A|Imelda|RRIM 600|480||
July 2024|Calixtro|Joemar|A|Imelda|RRIM 600|480||
July 2024|Ayagina|Marcila|A|Imelda|RRIM 600|960||
July 2024|Emboto|Bienvinida|M.|Imelda|RRIM 600|960||
July 2024|Pelegrino|Janeth||Imelda|RRIM 600|144||
July 2024|Amalio|Lolina||Imelda|RRIM 600|1440||
July 2024|Amalio|Jeokie|A|Imelda|RRIM 600|480||
July 2024|Dote|Edna||Bun.Brook|RRIM 600|1920||300
July 2024|Bartolome|Fresco||Bun.Brook|RRIM 600|1560||450
July 2024|Flores|Fernando|S|Bun.Brook|RRIM 600|960||310
July 2024|Muring|Julieto|G|Bun.Brook|RRIM 600|960||
July 2024|Ildefonso|Elvie|L|Bun.Brook|RRIM 600|480||
July 2024|Espaňola|Ma.Fe Theresa|L|Bun.Brook|RRIM 600|360||
July 2024|Lapeňa|Eliza|A|Bun.Brook|RRIM 600|480||
July 2024|Espaňola|Nonilo|A|Bun.Brook|RRIM 600|480||
July 2024|Isod|Lily|A|Bun.Brook|RRIM 600|960||200
July 2024|Kusong|Cynthia|C|Bun.Brook|RRIM 600|1440||
July 2024|Conijar|Robert|P|Bun.Brook|RRIM 600|480||
July 2024|Josol|Charita|E|Bun.Brook|RRIM 600|960||100
July 2024|Butaslac|Jessielyn||San Teodoro|RRIM 600|1200||
July 2024|Tonlay|Lucy||San Teodoro|RRIM 600|2160||
July 2024|Baron|Dolce||San Teodoro|RRIM 600|2400||350
July 2024|Gilloagan|Diego||San Teodoro|RRIM 600|2304||
July 2024|Garsuta|Sergio|A|San Teodoro|RRIM 600|1440||
July 2024|Marapao|Seme Tortor||San Teodoro|RRIM 600|2880||500
January 2025|Walley|Juvy||San Andres|RRIM 600|480||200
January 2025|Walley|Mario||San Andres|RRIM 600|480||150
January 2025|Beato|Nemesio||San Andres|RRIM 600|480||
January 2025|Danguis|Merry Joy|Y.|San Andres|RRIM 600|240||100
January 2025|Yesan|Marlyn|D.|San Andres|RRIM 600|480||1500
January 2025|Zamora|Alberto||San Andres|RRIM 600|5760||2000
January 2025|Ibarra Sr.|Rene|D.|San Andres|RRIM 600|1440||
January 2025|Quilly|Ruel||San Andres|RRIM 600|384||130
January 2025|Quilly|Christopher||San Andres|RRIM 600|576||180
January 2025|Quilly|Roberth||San Andres|RRIM 600|720||150
January 2025|lumbab|Julieto||San Andres|RRIM 600|4800||1500
January 2025|Atibagos|Sonny||San Andres|RRIM 600|3840||500
January 2025|Sentillas|Elvira|N.|San Andres|RRIM 600|2880||500
January 2025|Parcon|Tata||San Andres|RRIM 600|480||600
January 2025|Cabilin|Gondielina||San Andres|RRIM 600|2880||150
January 2025|Zacarias|Arniel||San Andres|RRIM 600|960||
January 2025|Ursola|Nenita||San Andres|RRIM 600|480||
January 2025|Campo Jr.|Jose|T.|San Andres|RRIM 600|960||
January 2025|Campo Jr.|Marcelino|T..|San Andres|RRIM 600|2400||
January 2025|Macadaya|Eglen||San Andres|RRIM 600|960||
January 2025|Campo Jr.|Ramon|T.|San Andres|RRIM 600|960||
January 2025|Nuňez|Lucy||San Andres|RRIM 600|1920||
January 2025|Roilo|Liza|N.|San Andres|RRIM 600|4800||
January 2025|Chavit|Levy||San Andres|RRIM 600|1440||
January 2025|Meranda|Mario||San Andres|RRIM 600|1440||
January 2025|Delapeňa Sr|Ruelito|A.|San Andres|RRIM 600|2400||
January 2025|Petallar|Alberto|S.|San Andres|RRIM 600|864||
January 2025|Nalam|Niel||San Andres|RRIM 600|2400||
January 2025|Binguan|Rojama||San Andres|RRIM 600|960||
January 2025|Riparip|Mayonnete|T.|San Andres|RRIM 600|720||
January 2025|Awa|Laila||San Andres|RRIM 600|480||
January 2025|Tajanlangit|Danilo|M.|San Andres|RRIM 600|1440||
January 2025|Granada|Erlindo||San Andres|RRIM 600|960||
January 2025|Granada|Arjel||San Andres|RRIM 600|480||
January 2025|Villanueva|Thaddeus||San Andres|RRIM 600|1440||
January 2025|Rosendo|Camarin||San Andres|RRIM 600|3840||
January 2025|Belar|Ramon||San Andres|RRIM 600|480||
January 2025|Nuňez|Bartolome||San Andres|RRIM 600|480||
January 2025|Belar|Geraldo||San Andres|RRIM 600|480||
January 2025|Belar|Fausto||San Andres|RRIM 600|240||
January 2025|Muňoz|Rico||San Andres|RRIM 600|1440||
January 2025|Ayala|Henry||San Andres|RRIM 600|1440||
January 2025|Lacaza|Andres||San Andres|RRIM 600|240||
January 2025|Sahay|Nonoy||San Andres|RRIM 600|480||
January 2025|Dagdag|Sharon||San Andres|RRIM 600|15360||28480
January 2025|Sitoza|Marites||Consuelo|RRIM 600|360||90
January 2025|Delacrus|Judith|S.|Consuelo|RRIM 600|432||100
January 2025|Delacrus|Sherly|S.|Consuelo|RRIM 600|432||
January 2025|Silvano|Marivic||Consuelo|RRIM 600|432||
January 2025|Bermudes|Rome||Consuelo|RRIM 600|720||300
January 2025|Calido|Arman||Consuelo|RRIM 600|1296||500
January 2025|Sangrador|Jose||Consuelo|RRIM 600|1200||500
January 2025|Maloloy-on|Gregorio||Consuelo|RRIM 600|1920||150
January 2025|Arana|Melvar||Consuelo|RRIM 600|480||130
January 2025|Arana|Jimley||Consuelo|RRIM 600|720||
January 2025|Tabo-Tabo|Nelvin||Consuelo|RRIM 600|480||
January 2025|Dionero|Mary Jane||Consuelo|RRIM 600|480||
January 2025|Arana|Jandie||Consuelo|RRIM 600|480||110
January 2025|Dionero|Cristine||Consuelo|RRIM 600|480||210
January 2025|Cabriadas|Emmanuel||Consuelo|RRIM 600|960||
January 2025|Capistrano|Joy||Consuelo|RRIM 600|720||
January 2025|Galos|Wilma||Consuelo|RRIM 600|960||
January 2025|Medina|Gilda||Imelda|RRIM 600|480||
January 2025|Anno|Jovelyn|P|Imelda|RRIM 600|500||
January 2025|Alayon|Loreta|B|Imelda|RRIM 600|250||
January 2025|Alejo|Shirly|A|Imelda|RRIM 600|125||
January 2025|Anteg|Julio|P|Imelda|RRIM 600|125||
January 2025|Arquinez|Isagani|B|Imelda|RRIM 600|125||
January 2025|Arquinez|Ricardo|B|Imelda|RRIM 600|130||
January 2025|Badua|Francisca|M|Imelda|RRIM 600|250||
January 2025|Badua|Leonito|V|Imelda|RRIM 600|125||
January 2025|Badua|Rodrigo|V|Imelda|RRIM 600|125||
January 2025|Badua|Rubin|S|Imelda|RRIM 600|125||
January 2025|Bisoy|Felicidad|A|Imelda|RRIM 600|165||
January 2025|Bistoyong|Mavic|G|Imelda|RRIM 600|1050||
January 2025|Dawing|Rosemarie|L|Imelda|RRIM 600|375||
January 2025|Guitilin|David|P|Imelda|RRIM 600|375||
January 2025|Goloran|Jenisa|A|Imelda|RRIM 600|130||
January 2025|Guitelin|Benito|P|Imelda|RRIM 600|100||
January 2025|Lozano|Pelagia|G|Imelda|RRIM 600|250||
January 2025|Qulaman|Wilson|B|Imelda|RRIM 600|1500||
January 2025|Ragusta|Judith||Imelda|RRIM 600|1500||
January 2025|Qulaman|Cristie|L|Imelda|RRIM 600|875||
January 2025|Suarez|Evelyn|D.|Imelda|RRIM 600|500||
January 2025|Oroc|Erico|.|Imelda|RRIM 600|100||
January 2025|Amalio|Juliano||Imelda|RRIM 600|105||
January 2025|Genelza|Reneboy||Imelda|RRIM 600|125||
January 2025|Alamada|Lolita||Imelda|RRIM 600|140||
January 2025|Fuentes|Johna||Imelda|RRIM 600|125||
January 2025|Tudtud|Federico||Imelda|RRIM 600|125||
January 2025|Agustines|Julius||Imelda|RRIM 600|100||
January 2025|Abadingo|Jamilo||Imelda|RRIM 600|2500||
January 2025|Alipio|Radsne||Imelda|RRIM 600|250||
January 2025|Amalio|Richard||Imelda|RRIM 600|250||
January 2025|Ettagonde|Jostino||Imelda|RRIM 600|100||
January 2025|Guingge|Jerry||Imelda|RRIM 600|125||
January 2025|Liba|Rodrigo||Imelda|RRIM 600|100||
January 2025|Paculba|Antonio||Imelda|RRIM 600|100||
January 2025|Tadle|Juliano||Imelda|RRIM 600|125||
January 2025|Villarmino|Esteban||Imelda|RRIM 600|100||
January 2025|Guinge|Lovely||Imelda|RRIM 600|125||
January 2025|Cordin|Leonida||Imelda|RRIM 600|125||
January 2025|Fuentes|Marikris||Imelda|RRIM 600|125||
January 2025|Fuentes|Joande||Imelda|RRIM 600|125||
January 2025|Geraldes|Reneboy||Imelda|RRIM 600|125||
January 2025|Polinio|Berto||Imelda|RRIM 600|100||
January 2025|Donaire|Dodong||Imelda|RRIM 600|250||
January 2025|Alejo|Romulo|H|Imelda|RRIM 600|250||
January 2025|Josol|Raymond|A|Bun.Brook|RRIM 600|480||
January 2025|Josol|Geralyn|G|Bun.Brook|RRIM 600|960||150
January 2025|Josol|Jeffrey|B|Bun.Brook|RRIM 600|480||
January 2025|Simino|Edmon|A|Bun.Brook|RRIM 600|960||
January 2025|Buenaflor|Sheryn|A|Bun.Brook|RRIM 600|960||
January 2025|Buenaflor|Jesusa|S|Bun.Brook|RRIM 600|1440||
January 2025|Uy|elizabeth|I|Bun.Brook|RRIM 600|480||
January 2025|Penida|Erlinda|C|Bun.Brook|RRIM 600|3120||350
January 2025|Penida|Bryan|C|Bun.Brook|RRIM 600|1440||180
January 2025|Rondan|Roden||Bun.Brook|RRIM 600|960||200
January 2025|Elorde|Ednard Gil|B|Bun.Brook|RRIM 600|5280||
January 2025|Rasonabe|Reneboy||San Teodoro|RRIM 600|480||
January 2025|Baron|Michael||San Teodoro|RRIM 600|528||
January 2025|Casal|Rosita|C|San Teodoro|RRIM 600|720||
January 2025|Osico|Lolita||San Teodoro|RRIM 600|1440||300
January 2025|Ibarra|Roberto|C|San Teodoro|RRIM 600|1200||
January 2025|Puspos|Solomon||San Teodoro|RRIM 600|500||200
January 2025|Diacoma|||San Teodoro|RRIM 600|1500||500"""


def parse_example():
    df = pd.read_csv(
        io.StringIO(EXAMPLE_DATA), sep="|", header=None, dtype=str, keep_default_na=False,
        names=["period", "last", "first", "middle", "farm_barangay", "variety",
               "no_of_trees", "cropping_density", "market_volume"],
    )
    for col in ["no_of_trees", "cropping_density", "market_volume"]:
        df[col] = df[col].apply(_clean_number)
    df["annual_volume"] = np.nan
    df["intercrop"] = None
    df["monocrop"] = "yes"
    return df


def parse_excel(upload):
    xls = pd.ExcelFile(upload)
    skip_terms = ["Province", "Municipality", "Commodity", "No.", "Prepared", "Noted",
                  "MELYDEN", "ENDRENIE", "ENGR", "Last Name", "Sexual", "Asexual",
                  "Intercrop", "Monocrop", "Complete Name"]
    rows = []
    for sheet in xls.sheet_names:
        raw = pd.read_excel(upload, sheet_name=sheet, header=None)
        period = None
        for _, row in raw.iterrows():
            if len(row) < 24:
                continue
            first = row.iloc[0]
            if pd.isna(first):
                continue
            fs = str(first).strip()
            if fs.startswith("Period"):
                period = " ".join(fs.split(":")[-1].strip().split())
                continue
            if any(k.lower() in fs.lower() for k in skip_terms):
                continue
            if fs.startswith("="):
                continue
            try:
                int(float(fs))
            except ValueError:
                continue
            if not period:
                continue
            rows.append({
                "period": period,
                "last": _clean_text(row.iloc[2]),
                "first": _clean_text(row.iloc[3]),
                "middle": _clean_text(row.iloc[4]),
                "farm_barangay": _clean_text(row.iloc[7]),
                "variety": _clean_text(row.iloc[10]),
                "no_of_trees": _clean_number(row.iloc[13]),
                "annual_volume": _clean_number(row.iloc[15]),
                "intercrop": _clean_tick(row.iloc[17]),
                "monocrop": _clean_tick(row.iloc[18]),
                "cropping_density": _clean_number(row.iloc[20]),
                "market_volume": _clean_number(row.iloc[23]),
            })
    return pd.DataFrame(rows)


def parse_csv(upload):
    df = pd.read_csv(upload, dtype=str, keep_default_na=False)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    rename_map = {
        "last_name": "last", "given_name": "first", "middle_name": "middle",
        "barangay": "farm_barangay", "farm_address": "farm_barangay",
        "trees": "no_of_trees", "tree_count": "no_of_trees",
        "volume": "market_volume", "density": "cropping_density",
        "intercrop_flag": "intercrop", "monocrop_flag": "monocrop",
    }
    df = df.rename(columns=rename_map)
    for col in ["no_of_trees", "annual_volume", "cropping_density", "market_volume"]:
        df[col] = df[col].apply(_clean_number) if col in df.columns else np.nan
    for col in ["intercrop", "monocrop"]:
        df[col] = df[col].apply(_clean_tick) if col in df.columns else None
    return df


def clean_data(df):
    for col in ["last", "first", "middle", "farm_barangay", "variety"]:
        if col in df.columns:
            df[col] = df[col].apply(_clean_text)
    df["farm_barangay"] = df["farm_barangay"].apply(_normalize_barangay)
    df["variety"] = df["variety"].apply(_normalize_variety)
    n_before_brgy = len(df)
    df = df[df["farm_barangay"].isin(VALID_BARANGAYS)].copy()
    n_dropped_brgy = n_before_brgy - len(df)
    n_before_trees = len(df)
    df = df.dropna(subset=["no_of_trees"]).reset_index(drop=True)
    n_dropped_trees = n_before_trees - len(df)
    df["farmer_name"] = (
        df["last"].fillna("") + ", " + df["first"].fillna("") + " " + df["middle"].fillna("")
    ).str.replace(r"\s+", " ", regex=True).str.strip(", ")
    df["cropping_system"] = np.where(
        df["monocrop"].notna(), "Monocrop",
        np.where(df["intercrop"].notna(), "Intercrop", "Unknown"),
    )
    df["production_kg"] = df["market_volume"].fillna(df["annual_volume"])
    return df, {
        "dropped_barangay": n_dropped_brgy,
        "dropped_trees": n_dropped_trees,
        "kept": len(df),
    }


def build_features(frame):
    f = frame.dropna(subset=["production_kg", "no_of_trees"]).copy()
    f = f[f["production_kg"] > 0]
    if len(f) < 5:
        return f, None, None
    le_v, le_b, le_c = LabelEncoder(), LabelEncoder(), LabelEncoder()
    f["variety_enc"] = le_v.fit_transform(f["variety"].fillna("Unknown"))
    f["brgy_enc"] = le_b.fit_transform(f["farm_barangay"].fillna("Unknown"))
    f["cs_enc"] = le_c.fit_transform(f["cropping_system"])
    X = f[["no_of_trees", "variety_enc", "brgy_enc", "cs_enc", "cropping_density"]].fillna(0)
    y = f["production_kg"].values
    return f, X, y


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mask = y_true > 0
    mape = float(mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100) if mask.sum() > 0 else np.nan
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan
    return {"RMSE": rmse, "MAE": mae, "MAPE (%)": mape, "R2": r2, "MAPE_n": int(mask.sum())}


def forecast_all_methods(values, periods):
    """Return dict of method name -> projected next-period value."""
    vals = np.asarray(values, dtype=float)
    n = len(vals)
    last = float(vals[-1])
    results = {}

    results["Naive (last value)"] = last
    results["Mean of observed periods"] = float(vals.mean())

    if n >= 2 and vals[-2] > 0:
        ratio = vals[-1] / vals[-2]
        results["Growth-rate extrapolation"] = float(vals[-1] * ratio)
    else:
        results["Growth-rate extrapolation"] = np.nan

    if n >= 2:
        t_idx = np.arange(n).reshape(-1, 1)
        lr = LinearRegression().fit(t_idx, vals)
        results["Linear trend (TAF)"] = float(lr.predict([[n]])[0])
    else:
        results["Linear trend (TAF)"] = np.nan

    try:
        am = ARIMA(vals, order=(0, 1, 0)).fit()
        results["ARIMA (0,1,0)"] = float(am.forecast(steps=1)[0])
    except Exception:
        results["ARIMA (0,1,0)"] = last

    return results


def style_plot(fig, height=400):
    fig.update_layout(
        height=height, plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Plus Jakarta Sans", color="#0f172a", size=12),
        margin=dict(l=20, r=20, t=70, b=20),
        title=dict(y=0.95, x=0.5, xanchor="center", yanchor="top"),
        xaxis=dict(gridcolor="#e2e8f0", linecolor="#e2e8f0"),
        yaxis=dict(gridcolor="#e2e8f0", linecolor="#e2e8f0"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


C_PRIMARY = "#1e3a8a"
C_ACCENT = "#d97706"
C_TERTIARY = "#0ea5e9"

st.sidebar.markdown("""
<div style="padding: 4px 0 18px 0;">
    <p class="sidebar-title">Rubber Production</p>
    <p class="sidebar-sub">Bunawan, Agusan del Sur</p>
</div>
""", unsafe_allow_html=True)

uploaded = st.sidebar.file_uploader("Upload dataset (CSV or Excel)", type=["csv", "xlsx", "xls"])
use_example = st.sidebar.checkbox("Use example dataset (if no upload)", value=True)

if uploaded is not None:
    try:
        raw = parse_csv(uploaded) if uploaded.name.lower().endswith(".csv") else parse_excel(uploaded)
    except Exception as exc:
        st.error(f"Could not read the uploaded file: {exc}")
        st.stop()
elif use_example:
    raw = parse_example()
else:
    st.info("Upload a dataset in the sidebar, or enable the example dataset to view the analysis.")
    st.stop()

df, quality = clean_data(raw)

if len(df) == 0:
    st.warning("No usable records were found after cleaning and filtering.")
    st.stop()

st.sidebar.markdown("---")

periods_all = sorted(df["period"].dropna().unique())
barangays_all = sorted(df["farm_barangay"].dropna().unique())
varieties_all = sorted(df["variety"].dropna().unique())

sel_periods = st.sidebar.multiselect("Period", periods_all, default=periods_all)
sel_barangays = st.sidebar.multiselect("Barangay", barangays_all, default=barangays_all)
sel_varieties = st.sidebar.multiselect("Variety", varieties_all, default=varieties_all)

d = df[
    df["period"].isin(sel_periods)
    & df["farm_barangay"].isin(sel_barangays)
    & df["variety"].isin(sel_varieties)
].copy()

if len(d) == 0:
    st.warning("No records match the current filters.")
    st.stop()

tabs = st.tabs(["Production Profile", "Trends", "Variable Relationships", "Model Accuracy", "Forecast"])

with tabs[0]:
    n_farmers = int(d["farmer_name"].nunique())
    n_records = len(d)
    n_with_volume = int(d["production_kg"].notna().sum())
    total_trees = int(d["no_of_trees"].sum(skipna=True))
    total_prod = float(d["production_kg"].sum(skipna=True))
    avg_per_farmer = total_prod / n_farmers if n_farmers > 0 else np.nan
    avg_per_recorded = float(d["production_kg"].mean(skipna=True)) if n_with_volume > 0 else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='metric-card'><div class='metric-label'>Farmers</div><div class='metric-value'>{n_farmers:,}</div><div class='metric-sub'>{n_records} records</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><div class='metric-label'>Total trees</div><div class='metric-value'>{total_trees:,}</div><div class='metric-sub'>across records</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><div class='metric-label'>Total production</div><div class='metric-value'>{total_prod:,.0f} kg</div><div class='metric-sub'>{n_with_volume} with volume</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card'><div class='metric-label'>Avg per farmer</div><div class='metric-value'>{avg_per_farmer:,.1f} kg</div><div class='metric-sub'>total / farmer count</div></div>", unsafe_allow_html=True)

    st.markdown(f"<div class='quality'>Data quality: {quality['kept']} records retained; {quality['dropped_barangay']} dropped by barangay whitelist; {quality['dropped_trees']} dropped for missing tree count. Among the {n_records} retained, {n_with_volume} have a recorded market volume. The mean recorded volume is <b>{avg_per_recorded:,.1f} kg</b>; the total divided by farmer count is <b>{avg_per_farmer:,.1f} kg</b>. Both are shown because the two denominators answer different questions.</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.3, 1])
    with col1:
        p_agg = d.groupby("period", as_index=False).agg(
            production=("production_kg", "sum"), farmers=("farmer_name", "nunique"))
        fig = px.bar(p_agg, x="period", y="production", text_auto=".2s", color_discrete_sequence=[C_PRIMARY])
        fig.update_traces(textposition="outside")
        fig.update_layout(title="Total production per period", xaxis_title="", yaxis_title="Production (kg)")
        st.plotly_chart(style_plot(fig), use_container_width=True)
    with col2:
        var_counts = d[d["variety"] != "Unknown"]["variety"].value_counts().reset_index()
        var_counts.columns = ["variety", "count"]
        if len(var_counts) > 0:
            fig = px.pie(var_counts, names="variety", values="count", hole=0.55,
                         color_discrete_sequence=[C_PRIMARY, C_ACCENT, C_TERTIARY])
            fig.update_layout(title="Variety distribution")
            st.plotly_chart(style_plot(fig), use_container_width=True)

    b_agg = (d.groupby("farm_barangay", as_index=False)
             .agg(production=("production_kg", "sum"), farmers=("farmer_name", "nunique"))
             .sort_values("production", ascending=False))
    fig = px.bar(b_agg, x="farm_barangay", y="production", color="production",
                 color_continuous_scale=["#dbeafe", C_PRIMARY, "#1e293b"], text_auto=".2s")
    fig.update_layout(title="Production per barangay", xaxis_title="", yaxis_title="Production (kg)", coloraxis_showscale=False)
    st.plotly_chart(style_plot(fig), use_container_width=True)

    if len(b_agg) > 0:
        top = b_agg.iloc[0]
        st.markdown(f"<div class='note'>Highest producing barangay is <b>{top['farm_barangay']}</b> with {top['production']:,.0f} kg from {int(top['farmers'])} farmer records.</div>", unsafe_allow_html=True)

with tabs[1]:
    p_agg = (d.groupby("period", as_index=False)
             .agg(production=("production_kg", "sum"), trees=("no_of_trees", "sum"), farmers=("farmer_name", "nunique"))
             .sort_values("period"))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p_agg["period"], y=p_agg["production"], mode="lines+markers",
                             line=dict(color=C_PRIMARY, width=3), marker=dict(size=12, color=C_ACCENT),
                             fill="tozeroy", fillcolor="rgba(30,58,138,0.10)", name="Production (kg)"))
    fig.update_layout(title="Production direction across periods", xaxis_title="", yaxis_title="Production (kg)")
    st.plotly_chart(style_plot(fig, height=360), use_container_width=True)

    if len(p_agg) >= 2:
        first_v, last_v = p_agg["production"].iloc[0], p_agg["production"].iloc[-1]
        pct = ((last_v - first_v) / first_v * 100) if first_v > 0 else np.nan
        direction = "increase" if pct > 0 else ("decrease" if pct < 0 else "no change")
        st.markdown(f"<div class='note'>Aggregated production shows a {direction} of {abs(pct):.1f}% between {p_agg['period'].iloc[0]} and {p_agg['period'].iloc[-1]}. With only two observation periods, the seasonal component cannot be estimated.</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        cs = d.groupby("cropping_system", as_index=False)["production_kg"].sum()
        fig = px.bar(cs, x="cropping_system", y="production_kg", color="cropping_system",
                     color_discrete_map={"Monocrop": C_PRIMARY, "Intercrop": C_ACCENT, "Unknown": "#94a3b8"})
        fig.update_layout(title="Production by cropping system", xaxis_title="", yaxis_title="Production (kg)", showlegend=False)
        st.plotly_chart(style_plot(fig, height=340), use_container_width=True)
    with col2:
        sc = d.dropna(subset=["no_of_trees", "production_kg"])
        if len(sc) > 0:
            fig = px.scatter(sc, x="no_of_trees", y="production_kg", color="period",
                             color_discrete_sequence=[C_PRIMARY, C_ACCENT], opacity=0.7)
            fig.update_layout(title="Trees vs production", xaxis_title="Number of trees", yaxis_title="Production (kg)")
            st.plotly_chart(style_plot(fig, height=340), use_container_width=True)

    top10 = d.groupby("farm_barangay", as_index=False)["production_kg"].sum().sort_values("production_kg", ascending=False).head(10)
    fig = px.bar(top10, x="production_kg", y="farm_barangay", orientation="h", color="production_kg",
                 color_continuous_scale=["#dbeafe", C_PRIMARY, "#1e293b"])
    fig.update_layout(title="Top barangays by production", xaxis_title="Production (kg)", yaxis_title="",
                      coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
    st.plotly_chart(style_plot(fig, height=400), use_container_width=True)

with tabs[2]:
    st.markdown("### Production by variety")
    var_agg = (d.groupby("variety", as_index=False)
               .agg(production=("production_kg", "sum"), farmers=("farmer_name", "nunique"),
                    avg_yield=("production_kg", "mean")).sort_values("production", ascending=False))
    st.dataframe(var_agg.rename(columns={"variety": "Variety", "production": "Total production (kg)",
                                          "farmers": "Farmers", "avg_yield": "Average yield (kg)"})
                 .style.format({"Total production (kg)": "{:,.0f}", "Average yield (kg)": "{:,.1f}"}),
                 use_container_width=True, hide_index=True)

    st.markdown("### Production by cropping system")
    cs_agg = d.groupby("cropping_system", as_index=False).agg(
        production=("production_kg", "sum"), farmers=("farmer_name", "nunique"), avg_yield=("production_kg", "mean"))
    st.dataframe(cs_agg.rename(columns={"cropping_system": "Cropping system", "production": "Total production (kg)",
                                        "farmers": "Farmers", "avg_yield": "Average yield (kg)"})
                 .style.format({"Total production (kg)": "{:,.0f}", "Average yield (kg)": "{:,.1f}"}),
                 use_container_width=True, hide_index=True)

    st.markdown("### Production by barangay")
    brgy_agg = (d.groupby("farm_barangay", as_index=False)
                .agg(production=("production_kg", "sum"), farmers=("farmer_name", "nunique"),
                     avg_yield=("production_kg", "mean")).sort_values("production", ascending=False))
    st.dataframe(brgy_agg.rename(columns={"farm_barangay": "Barangay", "production": "Total production (kg)",
                                          "farmers": "Farmers", "avg_yield": "Average yield (kg)"})
                 .style.format({"Total production (kg)": "{:,.0f}", "Average yield (kg)": "{:,.1f}"}),
                 use_container_width=True, hide_index=True)

    corr_frame = d.dropna(subset=["no_of_trees", "production_kg", "cropping_density"])
    if len(corr_frame) > 2:
        corr_trees = corr_frame["no_of_trees"].corr(corr_frame["production_kg"])
        corr_dens = corr_frame["cropping_density"].corr(corr_frame["production_kg"])
        st.markdown(f"<div class='note'>Pearson correlation between number of trees and production is {corr_trees:.3f}. Correlation between cropping density and production is {corr_dens:.3f}.</div>", unsafe_allow_html=True)

    crosstab = pd.crosstab(d["farm_barangay"], d["cropping_system"])
    if crosstab.size > 0:
        st.markdown("### Barangay by cropping system")
        st.dataframe(crosstab, use_container_width=True)

with tabs[3]:
    feat, X, y = build_features(d)
    if X is None or len(feat) < 10:
        st.warning("Not enough records after filtering to fit the models. At least 10 records with positive production volume are required.")
        st.stop()

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf_pred = cross_val_predict(rf, X, y, cv=kf)
    rf_m = compute_metrics(y, rf_pred)

    lr = LinearRegression()
    lr_pred = cross_val_predict(lr, X, y, cv=kf)
    lr_m = compute_metrics(y, lr_pred)

    period_series = d.groupby("period", as_index=False)["production_kg"].sum().sort_values("period")
    arima_m = {"RMSE": np.nan, "MAE": np.nan, "MAPE (%)": np.nan, "R2": np.nan, "MAPE_n": 0}
    if len(period_series) >= 2:
        try:
            am = ARIMA(period_series["production_kg"].values, order=(0, 1, 0)).fit()
            fitted = am.fittedvalues
            actual_aligned = period_series["production_kg"].values[-len(fitted):]
            arima_m = compute_metrics(actual_aligned, fitted)
        except Exception:
            pass

    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f"<div class='metric-card'><div class='metric-label'>RMSE</div><div class='metric-value'>{rf_m['RMSE']:,.2f}</div><div class='metric-sub'>Random Forest</div></div>", unsafe_allow_html=True)
    m2.markdown(f"<div class='metric-card'><div class='metric-label'>MAE</div><div class='metric-value'>{rf_m['MAE']:,.2f}</div><div class='metric-sub'>Random Forest</div></div>", unsafe_allow_html=True)
    m3.markdown(f"<div class='metric-card'><div class='metric-label'>MAPE</div><div class='metric-value'>{rf_m['MAPE (%)']:,.2f}%</div><div class='metric-sub'>on {rf_m['MAPE_n']} non-zero rows</div></div>", unsafe_allow_html=True)
    m4.markdown(f"<div class='metric-card'><div class='metric-label'>R2</div><div class='metric-value'>{rf_m['R2']:.3f}</div><div class='metric-sub'>Random Forest</div></div>", unsafe_allow_html=True)

    metrics_df = pd.DataFrame([
        {"Model": "Random Forest", **{k: rf_m[k] for k in ["RMSE", "MAE", "MAPE (%)", "R2"]}},
        {"Model": "Linear Regression", **{k: lr_m[k] for k in ["RMSE", "MAE", "MAPE (%)", "R2"]}},
        {"Model": "ARIMA (0,1,0)", **{k: arima_m[k] for k in ["RMSE", "MAE", "MAPE (%)", "R2"]}},
    ])
    st.markdown("### Performance summary")
    st.dataframe(metrics_df.style.format({"RMSE": "{:,.2f}", "MAE": "{:,.2f}", "MAPE (%)": "{:,.2f}", "R2": "{:.3f}"}),
                 use_container_width=True, hide_index=True)

    long_metrics = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Value").dropna(subset=["Value"])
    fig = px.bar(long_metrics, x="Metric", y="Value", color="Model", barmode="group", text_auto=".2f",
                 color_discrete_map={"Random Forest": C_PRIMARY, "Linear Regression": C_TERTIARY, "ARIMA (0,1,0)": C_ACCENT})
    fig.update_layout(title="Metric comparison across models")
    st.plotly_chart(style_plot(fig, height=380), use_container_width=True)

    rf.fit(X, y)
    imp = pd.DataFrame({
        "Feature": ["No. of trees", "Variety", "Barangay", "Cropping system", "Cropping density"],
        "Importance": rf.feature_importances_,
    }).sort_values("Importance", ascending=True)
    fig = px.bar(imp, x="Importance", y="Feature", orientation="h", color="Importance",
                 color_continuous_scale=["#dbeafe", C_PRIMARY, "#1e293b"])
    fig.update_layout(title="Random Forest feature importance", coloraxis_showscale=False, yaxis_title="")
    st.plotly_chart(style_plot(fig, height=340), use_container_width=True)

    best = "Random Forest"
    if not np.isnan(arima_m["MAPE (%)"]) and arima_m["MAPE (%)"] < rf_m["MAPE (%)"]:
        best = "ARIMA"
    if not np.isnan(lr_m["MAPE (%)"]) and lr_m["MAPE (%)"] < min(rf_m["MAPE (%)"], arima_m["MAPE (%)"] if not np.isnan(arima_m["MAPE (%)"]) else np.inf):
        best = "Linear Regression"

    st.markdown(f"<div class='note'>Lowest MAPE is achieved by <b>{best}</b>. Random Forest and Linear Regression use 5-fold cross-validation on {len(feat)} farm-level records; MAPE is computed only on rows with positive actual production ({rf_m['MAPE_n']} records), since percentage error is undefined at zero. ARIMA is fit on {len(period_series)} aggregated period totals and its residual error reflects the two-point series limitation. R2 is reported alongside because MAPE is sensitive to small denominators.</div>", unsafe_allow_html=True)

    st.markdown(f"<div class='caveat'><b>Modeling caveat:</b> Random Forest and Linear Regression solve a <i>cross-sectional</i> prediction problem (given a farm profile, predict its production). ARIMA solves a <i>temporal</i> forecasting problem (given past period totals, predict the next). These are different tasks, so comparing their metrics directly is only partially meaningful. The Random Forest metrics are the trustworthy ones; the ARIMA row is shown for completeness.</div>", unsafe_allow_html=True)

with tabs[4]:
    period_series = (d.groupby("period", as_index=False)["production_kg"].sum()
                     .sort_values("period").reset_index(drop=True))

    if len(period_series) < 2:
        st.warning("At least two periods are required to produce a forecast.")
        st.stop()

    values = period_series["production_kg"].values
    periods = period_series["period"].tolist()
    last_value = float(values[-1])
    last_period = periods[-1]

    method_results = forecast_all_methods(values, periods)

    feat, X, y = build_features(d)
    rf_next = np.nan
    if X is not None and len(feat) >= 10:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        oof_pred = cross_val_predict(rf, X, y, cv=kf)
        rf_next = float(np.clip(oof_pred, 0, None).sum())

    st.markdown("### Projection by method")
    assumption_map = {
        "Naive (last value)": "Next period equals the last observed level; no trend, no seasonality.",
        "Mean of observed periods": "Next period equals the historical average; smooths both observations.",
        "Growth-rate extrapolation": "The percentage change between the two periods repeats.",
        "Linear trend (TAF)": "A straight line through the two periods continues linearly.",
        "ARIMA (0,1,0)": "Random walk; forecast equals the last observed value.",
    }
    rows = []
    for name, val in method_results.items():
        rows.append({"Method": name, "Projection (kg)": val, "Assumption": assumption_map.get(name, "")})
    methods_df = pd.DataFrame(rows)
    st.dataframe(methods_df.style.format({"Projection (kg)": "{:,.0f}"}),
                 use_container_width=True, hide_index=True)

    valid = [v for v in method_results.values() if not np.isnan(v)]
    if valid:
        st.markdown(f"<div class='note'>Time-series projections range from <b>{min(valid):,.0f} kg</b> to <b>{max(valid):,.0f} kg</b> across the five methods. The spread is the honest measure of uncertainty: with only {len(period_series)} observation periods, no method can resolve seasonality, and none can be validated by cross-validation on such a short series. Report the range, not a single point.</div>", unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=periods, y=values, mode="lines+markers", name="Actual",
                             line=dict(color=C_PRIMARY, width=4), marker=dict(size=14)))
    projection_colors = {
        "Naive (last value)": "#64748b",
        "Mean of observed periods": "#14b8a6",
        "Growth-rate extrapolation": "#8b5cf6",
        "Linear trend (TAF)": C_ACCENT,
        "ARIMA (0,1,0)": "#ef4444",
    }
    projection_dash = {
        "Naive (last value)": "longdash",
        "Mean of observed periods": "longdash",
        "Growth-rate extrapolation": "dashdot",
        "Linear trend (TAF)": "dash",
        "ARIMA (0,1,0)": "dot",
    }
    next_label = "Projected"
    for name, val in method_results.items():
        if np.isnan(val):
            continue
        fig.add_trace(go.Scatter(
            x=[last_period, next_label], y=[last_value, val],
            mode="lines+markers", name=name,
            line=dict(color=projection_colors[name], width=2, dash=projection_dash[name]),
            marker=dict(size=10),
        ))
    if not np.isnan(rf_next):
        fig.add_trace(go.Scatter(
            x=[next_label], y=[rf_next], mode="markers", name="RF out-of-fold total",
            marker=dict(size=18, color=C_TERTIARY, symbol="star", line=dict(width=2, color="white")),
        ))
    fig.update_layout(title="Production trajectory with all projection methods",
                      xaxis_title="", yaxis_title="Production (kg)")
    st.plotly_chart(style_plot(fig, height=500), use_container_width=True)

    trend_val = method_results["Linear trend (TAF)"]
    growth_val = method_results["Growth-rate extrapolation"]
    naive_val = method_results["Naive (last value)"]

    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='metric-card'><div class='metric-label'>Trend (TAF)</div><div class='metric-value'>{trend_val:,.0f} kg</div><div class='metric-sub'>linear continuation</div></div>", unsafe_allow_html=True)
    if not np.isnan(growth_val):
        c2.markdown(f"<div class='metric-card'><div class='metric-label'>Growth-rate</div><div class='metric-value'>{growth_val:,.0f} kg</div><div class='metric-sub'>same % change</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><div class='metric-label'>Naive</div><div class='metric-value'>{naive_val:,.0f} kg</div><div class='metric-sub'>last observed level</div></div>", unsafe_allow_html=True)

    if not np.isnan(rf_next):
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Random Forest out-of-fold total</div><div class='metric-value'>{rf_next:,.0f} kg</div><div class='metric-sub'>aggregated across current farm profiles, not a temporal forecast</div></div>", unsafe_allow_html=True)

    st.markdown(f"<div class='caveat'><b>Why multiple methods:</b> With only {len(period_series)} observation periods, no single time-series model can be statistically validated. ARIMA(0,1,0) collapses to a random walk. Linear trend is transparent but assumes the observed direction continues. Growth-rate and mean projections bracket the plausible range. The Random Forest value is different in kind: it aggregates out-of-fold predictions on current farm profiles and is not a temporal forecast. Report the range, not a single point, when citing results.</div>", unsafe_allow_html=True)
