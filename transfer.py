import math
import os
from collections import Counter
import pandas as pd
import re
import tldextract
from urllib.parse import urlparse


# =====================================
#  敏感字與品牌字 (UF12, UF13)
# =====================================
SENSITIVE_WORDS = [
    "login","update","validate","activate","secure","account","signin",
    "verify","confirm","webscr","bank","payment","ebay","paypal","alert"
]

BRAND_WORDS = [
    "google","facebook","apple","amazon","microsoft","paypal","wechat",
    "instagram","linkedin","yahoo","netflix","office","outlook","icloud",
    "dropbox","adobe","walmart","tiktok","twitter"
]

FREE_HOST_SUFFIXES = [
    "weebly.com", "web.app", "firebaseapp.com", "pages.dev", "workers.dev",
    "blogspot.com", "duckdns.org", "github.io",
]

INFRA_CDN_HINTS = [
    "trafficmanager", "cloudfront", "akamai", "fastly", "cloudflare",
    "azure", "aws", "gvt1", "appsflyersdk", "firebaseio",
]

SUSPICIOUS_TLDS = {
    "sbs", "top", "cfd", "icu", "xyz", "shop", "cc", "zip", "mov",
}

SHORTENER_PATTERNS = [
    "bit\\.ly","goo\\.gl","tinyurl","ow\\.ly","t\\.co","is\\.gd",
    "buff\\.ly","rebrand\\.ly","cutt\\.ly","shorte\\.st","shorturl","trib\\.al"
]


# =====================================
#  URL Feature Extractor (UF2–UF19)
# =====================================
def extract_url_features(url, force_https=False):

    # ⭐ 若未標示協定，先記錄後補上 http 以便解析
    has_scheme = "://" in url
    if not has_scheme:
        url = "http://" + url
    elif force_https:
        # 白名單強制改為 https（UF9 必定為 0）
        url = re.sub(r"^[a-zA-Z]+://", "https://", url)

    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    scheme = parsed.scheme
    path = parsed.path or ""
    full_url = url
    features = {}

    # UF2
    dots = hostname.count(".")
    if dots > 3: features["UF2"] = 1
    elif dots == 3: features["UF2"] = 0.5
    else: features["UF2"] = 0

    # UF3
    ipv4 = r"^\d{1,3}(\.\d{1,3}){3}$"
    ipv6 = r"(([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4})"
    features["UF3"] = 1 if re.match(ipv4, hostname) or re.match(ipv6, hostname) else 0

    # UF4
    features["UF4"] = 1 if "@" in full_url else 0

    # UF5
    L = len(full_url)
    if L < 75: features["UF5"] = 0
    elif 75 <= L < 100: features["UF5"] = 0.5
    else: features["UF5"] = 1

    # UF6
    features["UF6"] = len([p for p in path.split("/") if p])

    # UF7
    pos = full_url.find("//", 8)
    features["UF7"] = 1 if pos != -1 else 0

    # UF8
    features["UF8"] = 1 if ("http" in hostname or "https" in hostname) else 0

    # ⭐ UF9：
    # https → 0
    # 非 https → 1
    # 無協定 → 0.5（未知）
    if not has_scheme:
        features["UF9"] = 0.5
    else:
        features["UF9"] = 0 if scheme == "https" else 1

    # UF10
    features["UF10"] = 1 if any(re.search(p, full_url, re.IGNORECASE) for p in SHORTENER_PATTERNS) else 0

    # UF11
    features["UF11"] = 1 if "-" in hostname else 0

    # UF12
    lower_url = full_url.lower()
    features["UF12"] = 1 if any(w in lower_url for w in SENSITIVE_WORDS) else 0

    # UF13
    features["UF13"] = 1 if any(w in lower_url for w in BRAND_WORDS) else 0

    # UF14
    features["UF14"] = 1 if any(c.isupper() for c in full_url) else 0

    # UF15
    features["UF15"] = 1 if full_url.count(".") > 2 else 0

    # UF16 - URL Entropy
    if len(full_url) == 0:
        ent = 0.0
    else:
        prob = [n / len(full_url) for n in Counter(full_url).values()]
        ent = -sum(p * math.log2(p) for p in prob)

    if ent < 3.5:
        features["UF16"] = 0
    elif ent <= 4.5:
        features["UF16"] = 1
    else:
        features["UF16"] = 2

    # UF17 - Query parameter count bucket
    if parsed.query:
        param_count = parsed.query.count("&") + 1
        if param_count <= 3:
            features["UF17"] = 1
        else:
            features["UF17"] = 2
    else:
        features["UF17"] = 0

    # UF18 - URL encoding
    features["UF18"] = 1 if "%" in full_url else 0

    # UF19 - Repetition ratio
    if len(full_url) == 0:
        features["UF19"] = 0.0
    else:
        counts = Counter(full_url)
        features["UF19"] = max(counts.values()) / len(full_url)

    extracted = tldextract.extract(full_url)
    subdomain = extracted.subdomain or ""
    domain = extracted.domain or ""
    suffix = extracted.suffix or ""
    registered = f"{domain}.{suffix}" if suffix else domain

    sub_lower = subdomain.lower()
    reg_lower = registered.lower()

    brand_in_sub = any(w in sub_lower for w in BRAND_WORDS)
    brand_in_reg = any(w in reg_lower for w in BRAND_WORDS)

    features["UF20"] = 1 if (brand_in_sub and not brand_in_reg) else 0
    features["UF21"] = 1 if brand_in_reg else 0

    features["UF22"] = len(registered)

    if len(registered) == 0:
        ent_reg = 0.0
    else:
        reg_prob = [n / len(registered) for n in Counter(registered).values()]
        ent_reg = -sum(p * math.log2(p) for p in reg_prob)

    if ent_reg < 2.5:
        features["UF23"] = 0
    elif ent_reg <= 3.5:
        features["UF23"] = 1
    else:
        features["UF23"] = 2

    features["UF24"] = 1 if any(registered.endswith(suf) for suf in FREE_HOST_SUFFIXES) else 0
    features["UF25"] = 1 if any(h in hostname for h in INFRA_CDN_HINTS) else 0
    features["UF26"] = 1 if suffix in SUSPICIOUS_TLDS else 0

    return features



if __name__ == "__main__":
    # =====================================
    #  Load datasets
    # =====================================
    base_dir = os.path.dirname(__file__)
    only_one_path = os.path.join(base_dir, "only_one.csv")
    top1m_path = os.path.join(base_dir, "top-1m.csv")

    df_black = pd.read_csv(only_one_path, header=None, names=["url"])
    df_white = pd.read_csv(top1m_path, header=None, names=["url"])

    df_black["label"] = 1
    df_white["label"] = 0

    # =====================================
    #  Feature extraction
    # =====================================
    def add_features(df, is_white=False):
        rows = []
        for url in df["url"]:
            rows.append(extract_url_features(url, force_https=is_white))
        return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

    df_black = add_features(df_black, is_white=False)
    df_white = add_features(df_white, is_white=True)  # 白名單一律 https → UF9=0

    # =====================================
    #  Output CSV
    # =====================================
    output = base_dir

    df_black.to_csv(os.path.join(output, "black_result.csv"), index=False)
    df_white.to_csv(os.path.join(output, "white_result.csv"), index=False)

    final_df = pd.concat([df_black, df_white], ignore_index=True)
    final_df.to_csv(os.path.join(output, "final_dataset.csv"), index=False)

    print("✔ 已完成輸出！")
