import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
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
.metric-card { background: #ffffff; border-radius: 12px; padding: 20px 22px; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(15,23,42,0.05); border-top: 3px solid #1e3a8a; }
.metric-label { color: #64748b; font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; }
.metric-value { color: #0f172a; font-size: 1.65rem; font-weight: 800; margin-top: 6px; }
.metric-sub { color: #d97706; font-size: 0.78rem; font-weight: 600; margin-top: 4px; }
.note { background: #fffbeb; border-left: 3px solid #d97706; padding: 14px 18px; border-radius: 6px; color: #78350f; font-size: 0.87rem; line-height: 1.6; margin-top: 14px; }
.sidebar-title { color: #1e3a8a; margin: 0; font-size: 1.05rem; font-weight: 800; }
.sidebar-sub { color: #64748b; font-size: 0.72rem; margin-top: 2px; }
.stTabs [data-baseweb="tab-list"] { gap: 10px; background: transparent; border-bottom: none; padding: 4px 0; }
.stTabs [data-baseweb="tab"] { background: #ffffff; border: 2px solid #e2e8f0; border-radius: 10px; padding: 10px 22px; color: #475569; font-weight: 600; font-size: 0.92rem; }
.stTabs [data-baseweb="tab"]:hover { border-color: #1e3a8a; color: #1e3a8a; }
.stTabs [aria-selected="true"] { background: #1e3a8a !important; color: #ffffff !important; border-color: #1e3a8a !important; }
.stTabs [data-baseweb="tab-highlight"] { display: none; }
.stTabs [data-baseweb="tab-border"] { display: none; }
</style>
""", unsafe_allow_html=True)

VALID_BARANGAYS = {"San Andres", "Consuelo", "Imelda", "Bun.Brook", "San Teodoro"}
VALID_VARIETIES = {"RRIM 600", "PB 280"}
BARANGAY_ALIASES = {"bun.brrook": "Bun.Brook", "santeodoro": "San Teodoro", "sanandres": "San Andres", "bunbrook": "Bun.Brook"}

def _clean_text(val):
    if pd.isna(val):
        return None
    s = str(val).strip()
    s = " ".join(s.split())
    s = "".join(ch for ch in s if ch.isprintable())
    if s == "" or s.lower() in {"nan", "none", "n/a", "/", "-"}:
        return None
    return s

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
    key = name.strip().lower().replace(" ", "")
    return BARANGAY_ALIASES.get(key, name.strip())

EXAMPLE_DATA = """July 2024|Sahay|Jerry||San Andres|RRIM 600|480|180|2000
July 2024|Sagrado|Improso||San Andres|RRIM 600|35000|4500|5000
July 2024|Roilo|Sheena|N.|San Andres|RRIM 600|480||
July 2024|Sanggo|Jerson||San Andres|RRIM 600|960||
July 2024|Ayangod|Carlina|D.|San Andres|RRIM 600|960||
July 2024|Dacuag|Carmelita||San Andres|RRIM 600|480||
July 2024|Gantalao|Corry|D.|San Andres|RRIM 600|480||
July 2024|Dicang|Dormita|B|San Andres|RRIM 600|720||
July 2024|Rein|Jason||San Andres|RRIM 600|480|180|200
July 2024|Jimenez|Danilo||San Andres|RRIM 600|1440|450|500
July 2024|Ayangod|Semon||San Andres|RRIM 600|720||
July 2024|Sabangan|Litto||San Andres|RRIM 600|480||
July 2024|Ocat|Juliet||San Andres|RRIM 600|240||
July 2024|Dela Cruz|Ludivina||San Andres|RRIM 600|240||
July 2024|Ayangod|kim||San Andres|RRIM 600|720||
July 2024|Banno|Domingo||San Andres|RRIM 600|480|95|100
July 2024|Mangibin|Leaster John||San Andres|RRIM 600|480||
July 2024|Mollaneda|Rona Joy||San Andres|RRIM 600|480|120|110
July 2024|Cayabas|Martina||San Andres|RRIM 600|480|105|90
July 2024|Ayangod|Myrna||San Andres|RRIM 600|960||
July 2024|Ayangod|Joeredella|T.|San Andres|RRIM 600|240||
July 2024|Mateo|Noel||San Andres|RRIM 600|720||
July 2024|Mateo|Caridad||San Andres|RRIM 600|720||
July 2024|Banno|Emmanuel||San Andres|RRIM 600|720|100|110
July 2024|Gula|Efren|M|San Andres|RRIM 600|240|95|80
July 2024|Magoncia|Jose|R.|San Andres|RRIM 600|240|120|105
July 2024|Ocat|Ronel|S.|San Andres|RRIM 600|240|85|70
July 2024|Crodua|charigine|E.|San Andres|RRIM 600|480|90|100
July 2024|Valdez|Benjamin|Q.|San Andres|RRIM 600|240|100|90
July 2024|Sentillas|Gilbert|N.|San Andres|RRIM 600|960|290|300
July 2024|Roilo|Jhon Lerenz|N.|San Andres|RRIM 600|1680|600|700
July 2024|Roilo|Henry|J.|San Andres|RRIM 600|1440|270|350
July 2024|Roda|Jerry|C.|San Andres|RRIM 600|480||
July 2024|Roda|Rodito Jr|C.|San Andres|RRIM 600|480|130|150
July 2024|Roda|Dennis|C.|San Andres|RRIM 600|480|105|120
July 2024|Escototo|Jesoniel|C.|San Andres|RRIM 600|480||
July 2024|Mangibin|Emily|M.|San Andres|RRIM 600|480|300|500
July 2024|Claro|Pepsinilla|T.|San Andres|RRIM 600|1440|450|600
July 2024|Salas|Candelario|H.|San Andres|RRIM 600|1440|530|350
July 2024|Palagan|Mary Ann|D.|San Andres|RRIM 600|720|230|260
July 2024|Bisoy|Norjon|D.|San Andres|RRIM 600|480|270|300
July 2024|Domingo|Danilo|P.|San Andres|RRIM 600|720|180|200
July 2024|Magoncia|Roberto 111|C.|San Andres|RRIM 600|960|120|150
July 2024|Walley|Marlon||San Andres|RRIM 600|480|160|180
July 2024|Galagala|Narciso||Consuelo|PB 280|1920|150|100
July 2024|Manog|Rolando||Consuelo|RRIM 600|240||
July 2024|Lozada|Melchor||Consuelo|RRIM 600|240|150|100
July 2024|Buenaflor|Herbert||Consuelo|RRIM 600|1440|350|400
July 2024|Galagala|Joseph||Consuelo|PB 280|960|300|350
July 2024|Galagala|Mamerto||Consuelo|PB 280|960|120|100
July 2024|Duhaylungsod|Sary||Consuelo|RRIM 600|480|140|900
July 2024|Mique|Arsenio||Consuelo|RRIM 600|480|250|300
July 2024|Galagala|Emelio||Consuelo|PB 280|1440|200|230
July 2024|Osic|Ronnie||Consuelo|RRIM 600|720|1300|100
July 2024|Galos|Simplicio||Consuelo|RRIM 600|480||
July 2024|Escatron|Elden||Consuelo|RRIM 600|480|1200|1400
July 2024|Blagantio|Ruth|I.|Consuelo|RRIM 600|6240|100|180
July 2024|Undag|Malaquias||Consuelo|RRIM 600|720||
July 2024|Casinto|Rolando||Consuelo|RRIM 600|360||
July 2024|Ramos|Alma||Consuelo|RRIM 600|360|100|80
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
January 2025|Walley|Juvy||San Andres|RRIM 600|480|180|200
January 2025|Walley|Mario||San Andres|RRIM 600|480|135|150
January 2025|Beato|Nemesio||San Andres|RRIM 600|480||
January 2025|Danguis|Merry Joy|Y.|San Andres|RRIM 600|240|95|100
January 2025|Yesan|Marlyn|D.|San Andres|RRIM 600|480|370|1500
January 2025|Zamora|Alberto||San Andres|RRIM 600|5760|1800|2000
January 2025|Ibarra Sr.|Rene|D.|San Andres|RRIM 600|1440||
January 2025|Quilly|Ruel||San Andres|RRIM 600|384|140|130
January 2025|Quilly|Christopher||San Andres|RRIM 600|576|210|180
January 2025|Quilly|Roberth||San Andres|RRIM 600|720|200|150
January 2025|lumbab|Julieto||San Andres|RRIM 600|4800|1200|1500
January 2025|Atibagos|Sonny||San Andres|RRIM 600|3840|650|500
January 2025|Sentillas|Elvira|N.|San Andres|RRIM 600|2880|380|500
January 2025|Parcon|Tata||San Andres|RRIM 600|480|350|600
January 2025|Cabilin|Gondielina||San Andres|RRIM 600|2880|210|150
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
January 2025|Sitoza|Marites||Consuelo|RRIM 600|360|105|90
January 2025|Delacrus|Judith|S.|Consuelo|RRIM 600|432|110|100
January 2025|Delacrus|Sherly|S.|Consuelo|RRIM 600|432||
January 2025|Silvano|Marivic||Consuelo|RRIM 600|432||
January 2025|Bermudes|Rome||Consuelo|RRIM 600|720|280|300
January 2025|Calido|Arman||Consuelo|RRIM 600|1296|430|500
January 2025|Sangrador|Jose||Consuelo|RRIM 600|1200|450|500
January 2025|Maloloy-on|Gregorio||Consuelo|RRIM 600|1920|110|150
January 2025|Arana|Melvar||Consuelo|RRIM 600|480|100|130
January 2025|Arana|Jimley||Consuelo|RRIM 600|720||
January 2025|Tabo-Tabo|Nelvin||Consuelo|RRIM 600|480||
January 2025|Dionero|Mary Jane||Consuelo|RRIM 600|480||
January 2025|Arana|Jandie||Consuelo|RRIM 600|480|100|110
January 2025|Dionero|Cristine||Consuelo|RRIM 600|480|200|210
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
        io.StringIO(EXAMPLE_DATA),
        sep="|",
        header=None,
        names=["period", "last", "first", "middle", "farm_barangay", "variety",
               "no_of_trees", "cropping_density", "market_volume"],
        dtype=str,
        keep_default_na=False,
    )
    for col in ["no_of_trees", "cropping_density", "market_volume"]:
        df[col] = df[col].apply(_clean_number)
    df["annual_volume"] = np.nan
    df["intercrop"] = None
    df["monocrop"] = None
    df["buyer_name"] = None
    df["buyer_location"] = None
    return df

def parse_uploaded_excel(upload):
    xls = pd.ExcelFile(upload)
    rows = []
    skip_terms = ["Province", "Municipality", "Commodity", "No.", "Prepared", "Noted",
                  "MELYDEN", "ENDRENIE", "ENGR", "Last Name", "Sexual", "Asexual",
                  "Intercrop", "Monocrop", "Complete Name"]
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
                "intercrop": _clean_text(row.iloc[17]),
                "monocrop": _clean_text(row.iloc[18]),
                "cropping_density": _clean_number(row.iloc[20]),
                "buyer_name": _clean_text(row.iloc[21]),
                "buyer_location": _clean_text(row.iloc[22]),
                "market_volume": _clean_number(row.iloc[23]),
            })
    return pd.DataFrame(rows)

def parse_uploaded_csv(upload):
    df = pd.read_csv(upload, dtype=str, keep_default_na=False)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    rename_map = {
        "last_name": "last", "given_name": "first", "middle_name": "middle",
        "farmer_last": "last", "farmer_first": "first",
        "barangay": "farm_barangay", "farm_address": "farm_barangay",
        "trees": "no_of_trees", "tree_count": "no_of_trees",
        "volume": "market_volume", "market_vol": "market_volume",
        "density": "cropping_density",
    }
    df = df.rename(columns=rename_map)
    for col in ["no_of_trees", "annual_volume", "cropping_density", "market_volume"]:
        if col in df.columns:
            df[col] = df[col].apply(_clean_number)
        else:
            df[col] = np.nan
    for col in ["intercrop", "monocrop", "buyer_name", "buyer_location"]:
        if col not in df.columns:
            df[col] = None
    return df

def clean_data(df):
    for col in ["last", "first", "middle", "farm_barangay", "variety",
                "intercrop", "monocrop", "buyer_name", "buyer_location"]:
        if col in df.columns:
            df[col] = df[col].apply(_clean_text)
    df["farm_barangay"] = df["farm_barangay"].apply(_normalize_barangay)
    df["variety"] = df["variety"].where(df["variety"].isin(VALID_VARIETIES), "Unknown")
    df = df[df["farm_barangay"].isin(VALID_BARANGAYS)].copy()
    df = df.dropna(subset=["no_of_trees"]).reset_index(drop=True)
    df["farmer_name"] = (
        df["last"].fillna("") + ", " + df["first"].fillna("") + " " + df["middle"].fillna("")
    ).str.replace(r"\s+", " ", regex=True).str.strip(", ")
    df["cropping_system"] = np.where(
        df["monocrop"].notna(), "Monocrop",
        np.where(df["intercrop"].notna(), "Intercrop", "Unknown")
    )
    df["production_kg"] = df["market_volume"].fillna(df["annual_volume"])
    return df

def build_features(frame):
    f = frame.dropna(subset=["production_kg", "no_of_trees"]).copy()
    f = f[f["production_kg"] > 0]
    if len(f) < 5:
        return f, None, None
    le_v = LabelEncoder()
    le_b = LabelEncoder()
    le_c = LabelEncoder()
    f["variety_enc"] = le_v.fit_transform(f["variety"].fillna("Unknown"))
    f["brgy_enc"] = le_b.fit_transform(f["farm_barangay"].fillna("Unknown"))
    f["cs_enc"] = le_c.fit_transform(f["cropping_system"])
    X = f[["no_of_trees", "variety_enc", "brgy_enc", "cs_enc", "cropping_density"]].fillna(0)
    y = f["production_kg"].values
    return f, X, y

def style_plot(fig, height=400):
    fig.update_layout(
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
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
C_GRID = "#e2e8f0"

st.sidebar.markdown("""
<div style="padding: 4px 0 18px 0;">
    <p class="sidebar-title">Rubber Production</p>
    <p class="sidebar-sub">Bunawan, Agusan del Sur</p>
</div>
""", unsafe_allow_html=True)

uploaded = st.sidebar.file_uploader("Upload dataset (CSV or Excel)", type=["csv", "xlsx", "xls"])
use_example = st.sidebar.checkbox("Use example dataset (if no upload)", value=True)

df_raw = None
if uploaded is not None:
    try:
        if uploaded.name.lower().endswith(".csv"):
            df_raw = parse_uploaded_csv(uploaded)
        else:
            df_raw = parse_uploaded_excel(uploaded)
    except Exception as exc:
        st.error(f"Could not read the uploaded file: {exc}")
        st.stop()
elif use_example:
    df_raw = parse_example()
else:
    st.info("Upload a dataset in the sidebar, or enable the example dataset to view the analysis.")
    st.stop()

df = clean_data(df_raw)

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

tabs = st.tabs([
    "Production Profile",
    "Trends",
    "Variable Relationships",
    "Model Accuracy",
    "Forecast",
])

with tabs[0]:
    total_farmers = int(d["farmer_name"].nunique())
    total_trees = int(d["no_of_trees"].sum(skipna=True))
    total_prod = float(d["production_kg"].sum(skipna=True))
    avg_yield = float(d["production_kg"].mean(skipna=True))
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "Farmers", f"{total_farmers:,}", "unique records"),
        (c2, "Total trees", f"{total_trees:,}", "across selected records"),
        (c3, "Total production", f"{total_prod:,.0f} kg", "sum of market volume"),
        (c4, "Average yield", f"{avg_yield:,.1f} kg", "per farmer record"),
    ]
    for col, label, val, sub in cards:
        col.markdown(
            f"<div class='metric-card'><div class='metric-label'>{label}</div>"
            f"<div class='metric-value'>{val}</div><div class='metric-sub'>{sub}</div></div>",
            unsafe_allow_html=True,
        )

    col1, col2 = st.columns([1.3, 1])
    with col1:
        p_agg = d.groupby("period", as_index=False).agg(
            production=("production_kg", "sum"),
            farmers=("farmer_name", "nunique"),
        )
        fig = px.bar(
            p_agg, x="period", y="production", text_auto=".2s",
            color_discrete_sequence=[C_PRIMARY],
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(title="Total production per period", xaxis_title="", yaxis_title="Production (kg)")
        st.plotly_chart(style_plot(fig), use_container_width=True)
    with col2:
        var_counts = d[d["variety"] != "Unknown"]["variety"].value_counts().reset_index()
        var_counts.columns = ["variety", "count"]
        if len(var_counts) > 0:
            fig = px.pie(
                var_counts, names="variety", values="count", hole=0.55,
                color_discrete_sequence=[C_PRIMARY, C_ACCENT, C_TERTIARY],
            )
            fig.update_layout(title="Variety distribution")
            st.plotly_chart(style_plot(fig), use_container_width=True)

    b_agg = (
        d.groupby("farm_barangay", as_index=False)
        .agg(production=("production_kg", "sum"), farmers=("farmer_name", "nunique"))
        .sort_values("production", ascending=False)
    )
    fig = px.bar(
        b_agg, x="farm_barangay", y="production", color="production",
        color_continuous_scale=["#dbeafe", C_PRIMARY, "#1e293b"], text_auto=".2s",
    )
    fig.update_layout(
        title="Production per barangay", xaxis_title="", yaxis_title="Production (kg)",
        coloraxis_showscale=False,
    )
    st.plotly_chart(style_plot(fig), use_container_width=True)

    if len(b_agg) > 0:
        top_brgy = b_agg.iloc[0]
        st.markdown(
            f"<div class='note'>Highest producing barangay is <b>{top_brgy['farm_barangay']}</b> "
            f"with {top_brgy['production']:,.0f} kg from {int(top_brgy['farmers'])} farmer records.</div>",
            unsafe_allow_html=True,
        )

with tabs[1]:
    p_agg = (
        d.groupby("period", as_index=False)
        .agg(
            production=("production_kg", "sum"),
            trees=("no_of_trees", "sum"),
            farmers=("farmer_name", "nunique"),
        )
        .sort_values("period")
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=p_agg["period"], y=p_agg["production"], mode="lines+markers",
            line=dict(color=C_PRIMARY, width=3),
            marker=dict(size=12, color=C_ACCENT),
            fill="tozeroy", fillcolor="rgba(30,58,138,0.10)",
            name="Production (kg)",
        )
    )
    fig.update_layout(title="Production direction across periods", xaxis_title="", yaxis_title="Production (kg)")
    st.plotly_chart(style_plot(fig, height=360), use_container_width=True)

    if len(p_agg) >= 2:
        first = p_agg["production"].iloc[0]
        last = p_agg["production"].iloc[-1]
        pct = ((last - first) / first * 100) if first > 0 else np.nan
        direction = "increase" if pct > 0 else ("decrease" if pct < 0 else "no change")
        st.markdown(
            f"<div class='note'>Aggregated production shows a {direction} of {abs(pct):.1f}% "
            f"between {p_agg['period'].iloc[0]} and {p_agg['period'].iloc[-1]}. With only two "
            f"observation periods available, the seasonal component cannot be estimated; the chart "
            f"reports the observable direction of change.</div>",
            unsafe_allow_html=True,
        )

    col1, col2 = st.columns(2)
    with col1:
        cs = d.groupby("cropping_system", as_index=False)["production_kg"].sum()
        fig = px.bar(
            cs, x="cropping_system", y="production_kg", color="cropping_system",
            color_discrete_map={"Monocrop": C_PRIMARY, "Intercrop": C_ACCENT, "Unknown": "#94a3b8"},
        )
        fig.update_layout(
            title="Production by cropping system", xaxis_title="", yaxis_title="Production (kg)",
            showlegend=False,
        )
        st.plotly_chart(style_plot(fig, height=340), use_container_width=True)
    with col2:
        sc = d.dropna(subset=["no_of_trees", "production_kg"])
        if len(sc) > 0:
            fig = px.scatter(
                sc, x="no_of_trees", y="production_kg", color="period",
                color_discrete_sequence=[C_PRIMARY, C_ACCENT], opacity=0.7,
            )
            fig.update_layout(title="Trees vs production", xaxis_title="Number of trees", yaxis_title="Production (kg)")
            st.plotly_chart(style_plot(fig, height=340), use_container_width=True)

    top = (
        d.groupby("farm_barangay", as_index=False)["production_kg"].sum()
        .sort_values("production_kg", ascending=False).head(10)
    )
    fig = px.bar(
        top, x="production_kg", y="farm_barangay", orientation="h", color="production_kg",
        color_continuous_scale=["#dbeafe", C_PRIMARY, "#1e293b"],
    )
    fig.update_layout(
        title="Top barangays by production", xaxis_title="Production (kg)", yaxis_title="",
        coloraxis_showscale=False, yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(style_plot(fig, height=400), use_container_width=True)

with tabs[2]:
    st.markdown("### Production by variety")
    var_agg = (
        d.groupby("variety", as_index=False)
        .agg(
            production=("production_kg", "sum"),
            farmers=("farmer_name", "nunique"),
            avg_yield=("production_kg", "mean"),
        )
        .sort_values("production", ascending=False)
    )
    st.dataframe(
        var_agg.rename(columns={
            "variety": "Variety", "production": "Total production (kg)",
            "farmers": "Farmers", "avg_yield": "Average yield (kg)",
        }).style.format({
            "Total production (kg)": "{:,.0f}", "Average yield (kg)": "{:,.1f}",
        }),
        use_container_width=True, hide_index=True,
    )

    st.markdown("### Production by cropping system")
    cs_agg = d.groupby("cropping_system", as_index=False).agg(
        production=("production_kg", "sum"),
        farmers=("farmer_name", "nunique"),
        avg_yield=("production_kg", "mean"),
    )
    st.dataframe(
        cs_agg.rename(columns={
            "cropping_system": "Cropping system", "production": "Total production (kg)",
            "farmers": "Farmers", "avg_yield": "Average yield (kg)",
        }).style.format({
            "Total production (kg)": "{:,.0f}", "Average yield (kg)": "{:,.1f}",
        }),
        use_container_width=True, hide_index=True,
    )

    st.markdown("### Production by barangay")
    brgy_agg = (
        d.groupby("farm_barangay", as_index=False)
        .agg(
            production=("production_kg", "sum"),
            farmers=("farmer_name", "nunique"),
            avg_yield=("production_kg", "mean"),
        )
        .sort_values("production", ascending=False)
    )
    st.dataframe(
        brgy_agg.rename(columns={
            "farm_barangay": "Barangay", "production": "Total production (kg)",
            "farmers": "Farmers", "avg_yield": "Average yield (kg)",
        }).style.format({
            "Total production (kg)": "{:,.0f}", "Average yield (kg)": "{:,.1f}",
        }),
        use_container_width=True, hide_index=True,
    )

    corr_frame = d.dropna(subset=["no_of_trees", "production_kg", "cropping_density"])
    if len(corr_frame) > 2:
        corr_trees = corr_frame["no_of_trees"].corr(corr_frame["production_kg"])
        corr_dens = corr_frame["cropping_density"].corr(corr_frame["production_kg"])
        st.markdown(
            f"<div class='note'>Pearson correlation between number of trees and production is "
            f"{corr_trees:.3f}. Correlation between cropping density and production is {corr_dens:.3f}.</div>",
            unsafe_allow_html=True,
        )

    crosstab = pd.crosstab(d["farm_barangay"], d["cropping_system"])
    if crosstab.size > 0:
        st.markdown("### Barangay by cropping system")
        st.dataframe(crosstab, use_container_width=True)

with tabs[3]:
    feat, X, y = build_features(d)
    if X is None or len(feat) < 10:
        st.warning("Not enough records after filtering to fit the models.")
        st.stop()

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf_pred = cross_val_predict(rf, X, y, cv=kf)
    rf_rmse = float(np.sqrt(mean_squared_error(y, rf_pred)))
    rf_mae = float(mean_absolute_error(y, rf_pred))
    rf_mape = float(mean_absolute_percentage_error(y, rf_pred) * 100)

    lr = LinearRegression()
    lr_pred = cross_val_predict(lr, X, y, cv=kf)
    lr_rmse = float(np.sqrt(mean_squared_error(y, lr_pred)))
    lr_mae = float(mean_absolute_error(y, lr_pred))
    lr_mape = float(mean_absolute_percentage_error(y, lr_pred) * 100)

    period_series = d.groupby("period", as_index=False)["production_kg"].sum().sort_values("period")
    arima_rmse = arima_mae = arima_mape = np.nan
    if len(period_series) >= 2:
        try:
            am = ARIMA(period_series["production_kg"].values, order=(0, 1, 0))
            af = am.fit()
            fitted = af.fittedvalues
            actual_aligned = period_series["production_kg"].values[-len(fitted):]
            arima_rmse = float(np.sqrt(mean_squared_error(actual_aligned, fitted)))
            arima_mae = float(mean_absolute_error(actual_aligned, fitted))
            arima_mape = float(mean_absolute_percentage_error(actual_aligned, fitted) * 100)
        except Exception:
            pass

    m1, m2, m3 = st.columns(3)
    m1.markdown(
        f"<div class='metric-card'><div class='metric-label'>RMSE</div>"
        f"<div class='metric-value'>{rf_rmse:,.2f}</div><div class='metric-sub'>Random Forest</div></div>",
        unsafe_allow_html=True,
    )
    m2.markdown(
        f"<div class='metric-card'><div class='metric-label'>MAE</div>"
        f"<div class='metric-value'>{rf_mae:,.2f}</div><div class='metric-sub'>Random Forest</div></div>",
        unsafe_allow_html=True,
    )
    m3.markdown(
        f"<div class='metric-card'><div class='metric-label'>MAPE</div>"
        f"<div class='metric-value'>{rf_mape:,.2f}%</div><div class='metric-sub'>Random Forest</div></div>",
        unsafe_allow_html=True,
    )

    metrics_df = pd.DataFrame({
        "Model": ["Random Forest", "Linear Regression", "ARIMA"],
        "RMSE": [rf_rmse, lr_rmse, arima_rmse],
        "MAE": [rf_mae, lr_mae, arima_mae],
        "MAPE (%)": [rf_mape, lr_mape, arima_mape],
    })
    st.markdown("### Performance summary")
    st.dataframe(
        metrics_df.style.format({"RMSE": "{:,.2f}", "MAE": "{:,.2f}", "MAPE (%)": "{:,.2f}"}),
        use_container_width=True, hide_index=True,
    )

    long_metrics = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Value")
    fig = px.bar(
        long_metrics, x="Metric", y="Value", color="Model", barmode="group",
        text_auto=".2f",
        color_discrete_map={
            "Random Forest": C_PRIMARY, "Linear Regression": C_TERTIARY, "ARIMA": C_ACCENT,
        },
    )
    fig.update_layout(title="Metric comparison across models")
    st.plotly_chart(style_plot(fig, height=380), use_container_width=True)

    rf.fit(X, y)
    imp = pd.DataFrame({
        "Feature": ["No. of trees", "Variety", "Barangay", "Cropping system", "Cropping density"],
        "Importance": rf.feature_importances_,
    }).sort_values("Importance", ascending=True)
    fig = px.bar(
        imp, x="Importance", y="Feature", orientation="h", color="Importance",
        color_continuous_scale=["#dbeafe", C_PRIMARY, "#1e293b"],
    )
    fig.update_layout(
        title="Random Forest feature importance",
        coloraxis_showscale=False, yaxis_title="",
    )
    st.plotly_chart(style_plot(fig, height=340), use_container_width=True)

    best_label = "Random Forest"
    if not np.isnan(arima_mape) and arima_mape < rf_mape:
        best_label = "ARIMA"
    st.markdown(
        f"<div class='note'>Lowest MAPE is achieved by <b>{best_label}</b>. Random Forest is fit on "
        f"farm-level features with 5-fold cross-validation; ARIMA is fit on the aggregated period "
        f"totals and its residual error reflects the two-point series limitation.</div>",
        unsafe_allow_html=True,
    )

with tabs[4]:
    period_series = d.groupby("period", as_index=False)["production_kg"].sum().sort_values("period")
    if len(period_series) < 2:
        st.warning("At least two periods are required to produce a forecast.")
        st.stop()

    try:
        am = ARIMA(period_series["production_kg"].values, order=(0, 1, 0))
        af = am.fit()
        arima_next = float(af.forecast(steps=1)[0])
    except Exception:
        arima_next = float(period_series["production_kg"].iloc[-1])

    feat, X, y = build_features(d)
    rf_next = np.nan
    if X is not None and len(feat) >= 10:
        rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        rf_next = float(rf.predict(X).sum())

    last_label = period_series["period"].iloc[-1]
    next_label = "Projected next period"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=period_series["period"], y=period_series["production_kg"],
            mode="lines+markers", name="Actual",
            line=dict(color=C_PRIMARY, width=3), marker=dict(size=12),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[last_label, next_label],
            y=[period_series["production_kg"].iloc[-1], arima_next],
            mode="lines+markers", name="ARIMA projection",
            line=dict(color=C_ACCENT, width=3, dash="dash"),
            marker=dict(size=12, symbol="diamond"),
        )
    )
    if not np.isnan(rf_next):
        fig.add_trace(
            go.Scatter(
                x=[last_label, next_label],
                y=[period_series["production_kg"].iloc[-1], rf_next],
                mode="lines+markers", name="Random Forest projection",
                line=dict(color=C_TERTIARY, width=3, dash="dot"),
                marker=dict(size=12, symbol="square"),
            )
        )
    fig.update_layout(
        title="Production trajectory and next-period projection",
        xaxis_title="", yaxis_title="Production (kg)",
    )
    st.plotly_chart(style_plot(fig, height=440), use_container_width=True)

    c1, c2 = st.columns(2)
    c1.markdown(
        f"<div class='metric-card'><div class='metric-label'>ARIMA projected production</div>"
        f"<div class='metric-value'>{arima_next:,.0f} kg</div><div class='metric-sub'>next period</div></div>",
        unsafe_allow_html=True,
    )
    if not np.isnan(rf_next):
        c2.markdown(
            f"<div class='metric-card'><div class='metric-label'>Random Forest projected production</div>"
            f"<div class='metric-value'>{rf_next:,.0f} kg</div>"
            f"<div class='metric-sub'>aggregated over current farm profile</div></div>",
            unsafe_allow_html=True,
        )

    last_actual = float(period_series["production_kg"].iloc[-1])
    delta_arima = arima_next - last_actual
    direction_word = "increase" if delta_arima > 0 else ("decrease" if delta_arima < 0 else "no change")
    st.markdown(
        f"<div class='note'>ARIMA projection indicates a {direction_word} of {abs(delta_arima):,.0f} kg "
        f"relative to the latest observed period. The random-walk formulation reflects the two-point "
        f"series; the projection is a directional signal, not a precise point estimate.</div>",
        unsafe_allow_html=True,
    )
