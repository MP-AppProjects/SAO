# SAO — System Analiz Openfield

Webowa aplikacja analityczna w Pythonie, polska alternatywa dla SPSS. Przeznaczona dla analityków badań społecznych, ankieterskich i marketingowych.

---

## Wymagania

- Python 3.10
- Biblioteki z `requirements.txt`

```bash
pip install -r requirements.txt
```

---

## Uruchomienie lokalne

```bash
streamlit run generator.py
```

Aplikacja dostępna pod adresem: `http://localhost:8501`

---

## Pierwsze logowanie

| Login | Hasło |
|-------|-------|
| `admin` | `admin` |

Po pierwszym zalogowaniu system wymusi zmianę hasła.

---

## Funkcje

### Moduły analityczne
- **Przygotowanie danych** — braki danych, etykiety, rekodowanie, grupowanie, ważenie (RIM), podział na podzbiory (Split File)
- **Analizy i tabele** — częstości, tabele krzyżowe, tabele matrycowe, średnie, statystyki opisowe, korelacje
- **Regresja** — OLS i logistyczna
- **ANOVA**
- **Testy normalności**
- **Analiza czynnikowa**
- **Skupienia i segmentacja** — k-means, klastrowanie hierarchiczne
- **Conjoint**
- **MaxDiff**
- **Chmura słów**
- **Eksport** — do Excela (.xlsx) i PowerPoint (.pptx)

### Panel administracyjny
- Tworzenie i zarządzanie kontami użytkowników
- Nadawanie uprawnień per moduł
- Czasowy dostęp (data wygaśnięcia konta)
- Śledzenie aktywności użytkowników
- Historia logowań z geolokalizacją IP
- Podgląd aktywnych sesji i możliwość wymuszenia wylogowania
- Statystyki użycia
- Audit log

### Obsługiwane formaty plików
- SPSS (`.sav`)
- Excel (`.xlsx`, `.xls`)

---

## Baza danych

Aplikacja używa SQLite. Plik bazy tworzy się automatycznie przy pierwszym uruchomieniu:

```
data/sao_admin.db
```

Katalog `data/` tworzony jest automatycznie — nie wymaga ręcznej konfiguracji.

**Hasła** przechowywane są w formie skrótu PBKDF2-HMAC-SHA256 (260 000 iteracji) z losowym salt — nigdy w formie jawnej.

---

## Deployment na serwerze (Linux)

### 1. Instalacja zależności

```bash
apt update && apt install python3-pip python3-venv nginx -y
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Usługa systemd

Utwórz plik `/etc/systemd/system/sao.service`:

```ini
[Unit]
Description=SAO Streamlit App
After=network.target

[Service]
User=www-data
WorkingDirectory=/opt/sao
ExecStart=/opt/sao/venv/bin/streamlit run generator.py \
    --server.port 8501 \
    --server.address 127.0.0.1 \
    --server.headless true
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
systemctl daemon-reload
systemctl enable sao
systemctl start sao
```

### 3. Nginx — reverse proxy

Utwórz plik `/etc/nginx/sites-available/sao`:

```nginx
server {
    listen 80;
    server_name twoja-domena.pl;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

```bash
ln -s /etc/nginx/sites-available/sao /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx
```

### 4. HTTPS (Let's Encrypt)

```bash
apt install certbot python3-certbot-nginx -y
certbot --nginx -d twoja-domena.pl
```

---

## Backup bazy danych

Plik `data/sao_admin.db` zawiera wszystkich użytkowników, hasła (zahashowane) i logi aktywności. Należy go regularnie backupować:

```bash
cp data/sao_admin.db data/sao_admin_$(date +%Y%m%d).db
```

---

## Struktura projektu

```
SAO/
├── generator.py        # cała aplikacja (jeden plik)
├── requirements.txt    # zależności Python
├── README.md
└── data/
    └── sao_admin.db    # baza danych (tworzona automatycznie)
```
