# CLAUDE.md — Kontekst projektu SAO

Ten plik jest automatycznie czytany przez Claude Code przy każdej sesji w tym katalogu. Zawiera kluczowy kontekst projektu, który pozwoli Ci pracować bez zbędnego tłumaczenia.

---

## Czym jest ten projekt?

**SAO (System Analiz Openfield)** — webowa aplikacja analityczna w Pythonie, polska alternatywa dla SPSS, pisana w Streamlit. Służy analitykom badań społecznych, ankieterskich i marketingowych do obróbki i analizy danych z ankiet.

**Pliki źródłowe (3 moduły):**
- `generator.py` (~8400 linii) — ciało aplikacji Streamlit: gate logowania, inicjalizacja stanu, wczytywanie danych, pipeline przygotowania i całe UI (dispatch `elif menu == "..."`).
- `sao_admin.py` (~490 linii) — backend: uwierzytelnianie, baza SQLite, sesje, uprawnienia, geolokalizacja. Importowany przez `from sao_admin import *`.
- `sao_core.py` (~2440 linii) — warstwa analityczna i eksporty: wczytywanie SPSS/Excel, transformacje (rekodowania, czyszczenie, segmentacje), budowa tabel, statystyka (regresja, ANOVA, korelacje, czynnikowa, conjoint, maxdiff, wagi RIM), eksport do Excela. Importowany przez `import sao_core; from sao_core import *`.

Wszystkie trzy pliki muszą leżeć w tym samym katalogu — `streamlit run generator.py` działa bez zmian.

**Język UI:** polski (wszystkie komunikaty, etykiety, instrukcje po polsku).

**Rozmawiaj ze mną (użytkownikiem) po polsku.**

---

## Architektura — najważniejsze zasady

### Podział na moduły
Kod jest podzielony na 3 pliki (patrz wyżej). Zasada granic:
- **`sao_admin.py`** i **`sao_core.py`** zawierają wyłącznie czyste funkcje (bez UI Streamlita poza `st.session_state`/cache). Nie odwołują się do zmiennych globalnych z ciała skryptu.
- **`generator.py`** zawiera całe UI i wykonanie modułowe. Nowe funkcje pomocnicze (czyste, wielokrotnego użytku) dodawaj do `sao_core.py`; nowe widoki/UI — do `generator.py`.
- Wyjątek: `get_var_display_name` w `sao_core.py` potrzebuje zbioru `original_cols` z `generator.py` — jest on **wstrzykiwany** po wczytaniu danych przez `sao_core.original_cols = original_cols`. Jeśli przenosisz do `sao_core` funkcję czytającą globalną z `generator.py`, zastosuj ten sam wzorzec wstrzykiwania.
- Nie rozbijaj UI (`elif menu`) na osobne pliki bez wyraźnego polecenia — sekwencyjny przepływ stanu jest tam celowy.

### Wydajność — memoizacja pipeline'u danych
W `generator.py` pipeline przygotowania danych (kopie DataFrame, KMeans w segmentacjach, rekodowania, czyszczenie, regex braków) jest **memoizowany** na fingerprincie wejść (`_wd_key` / `_wd_cache` w `st.session_state`). Przelicza się ponownie tylko, gdy zmieni się któreś z wejść. Jeśli dodajesz nowy krok transformujący `df`/`df_raw` przy każdym rerun — dorzuć jego wejście do `_wd_payload`, inaczej cache nie wykryje zmiany.

### ASCII encoding
**Wszystkie trzy pliki źródłowe (`generator.py`, `sao_admin.py`, `sao_core.py`) MUSZĄ być zapisane w czystym ASCII.** Wszystkie polskie znaki i emoji są escape'owane jako `\uXXXX` albo `\UXXXXXXXX`. Np.:
- `"Cz\u0119sto\u015bci"` zamiast `"Częstości"`
- `"\U0001f500"` zamiast `"🔀"`
- `"\u2705"` zamiast `"✅"`

Uwaga: w `.py` escape `\uXXXX` działa w literałach stringów, ale **nie umieszczaj** literalnego `\uXXXX` w docstringach/komentarzach jako tekstu (Python parsuje docstring jako string i rzuci `truncated \uXXXX escape`).

Po każdej edycji sprawdź każdy zmieniony plik, np.: `python -c "open('sao_core.py','rb').read().decode('ascii')"` — jeśli nie rzuci błędu, plik jest OK. Jeśli wprowadzisz znaki Unicode bezpośrednio, po zapisaniu pliku przez innego użytkownika na Windows mogą się one zepsuć.

### Python 3.10 compatibility
Użytkownik uruchamia aplikację na **Python 3.10** w Windows. Ma to jedno kluczowe ograniczenie:

**f-stringi w Python 3.10 NIE pozwalają na backslash w części wyrażenia `{...}`.**

ŹLE:
```python
st.error(f"Grupa `{_grp_lbl or 'pe\u0142na baza'}`")  # BŁĄD!
```

DOBRZE:
```python
_grp_disp = _grp_lbl or 'pe\u0142na baza'
st.error(f"Grupa `{_grp_disp}`")
```

Jeśli potrzebujesz escape Unicode wewnątrz f-stringa — wyciągnij wartość do zmiennej przed f-stringiem.

### Streamlit-specific
- UI aplikacji to jeden długi blok w `generator.py` z rozgałęzieniami przez `elif menu == "..."` (nav sidebar); logika analityczna i backend są w `sao_core.py` / `sao_admin.py`
- Stan trzymany w `st.session_state` — lista kluczy w sekcji inicjalizacji
- Wyniki analiz kumulują się (helper `_merge_result(list, entry, key_fn)`) — nie nadpisują
- Każda analiza ma UI do usuwania wyników (🗑️ per wynik + "Usuń wszystkie")

---

## Menu aplikacji (14 modułów)

1. 🏠 Dashboard
2. 📁 Projekt i Słownik (wczytywanie SPSS/Excel, zapis projektu JSON)
3. 🛠️ Przygotowanie Danych — zakładki:
   - Braki danych
   - Etykiety (zmiennych i wartości)
   - Typy zmiennych
   - Czyszczenie
   - Pytania wielokrotne
   - Pytania matrycowe
   - Ważenie (RIM)
   - Rekodowanie
   - Grupowanie odpowiedzi
   - 🔀 Podział na podzbiory (SPSS Split File)
4. 📈 Analizy i Tabele — Częstości | Matrycowe | Krzyżowe | Średnie | Opisowe | Korelacje
5. 📐 Testy Normalności
6. 📉 Regresja (OLS + Logistyczna)
7. 📊 ANOVA
8. 🔬 Analiza Czynnikowa
9. 🎯 Skupienia i Segmentacja (w tym klastrowanie hierarchiczne)
10. 📊 Conjoint
11. 🔢 MaxDiff
12. ☁️ Chmura Słów
13. 💾 Eksport do Excela
14. 📊 Eksport do PowerPoint

---

## Kluczowe funkcje systemowe

### Split File (SPSS-style podział na podzbiory)
- `st.session_state.split_var` — nazwa zmiennej grupującej lub `None`
- Helper `_iter_split_groups(df, df_raw, var_labels, split_var, weights=None)` — yielduje `(group_label, df_slice, df_raw_slice, weights_slice)` dla każdej grupy; jeśli split wyłączony, yielduje jeden pustą etykietą
- Helper `_split_badge(grp_label)` — żółty info-box wewnątrz expandera
- Helper `_extract_split_from_title(key)` — parsuje `"Q1 | plec=Kobieta"` → `("Q1", "plec=Kobieta")`
- Wskaźnik aktywnego splita w `module_header()` — żółty pasek pod nagłówkiem
- Klucze wyników ze splitem: `"base_title | group=value"` (dict) lub pole `group_label` w entry (list)

### Ważenie (RIM / raking)
- `st.session_state.weights` — np.array lub `None`
- `st.session_state.weight_targets` — dict z celami ważenia
- Normalizacja SPSS-compatible: suma wag = N
- Wskaźnik aktywnych wag w `module_header()` — zielony pasek pod nagłówkiem
- Efektywna wielkość próby (ESS) używana do testów istotności przy wagach

### Kumulacja wyników
Wszystkie moduły używają helper'a:
```python
_merge_result(results_list, new_entry, key_fn)
```
który dodaje wpis lub zastępuje istniejący o tym samym kluczu. Klucz jest tworzony przez `key_fn(entry)` — np. `lambda r: (r['dep_var'], r.get('group_label',''))`. Przy splicie klucz MUSI zawierać `group_label`.

### Zapis/odczyt projektu JSON
Menu "Projekt i Słownik" ma przyciski zapisu/odczytu stanu całej sesji do/z pliku `.json`. Wszystko co dodajesz do session_state i co powinno przetrwać — dodaj do `data = {...}` w sekcji zapisu i do `st.session_state.XXX = raw_data.get(...)` w sekcji odczytu.

---

## Konwencje stylu kodu

- **Nazwy funkcji:** snake_case (`run_regression_block`, `apply_recodings`)
- **Zmienne pomocnicze w UI:** prefiks `_` (`_exp`, `_rc1`, `_grp_lbl`, `_ri`)
- **Session state keys:** bez prefixu (np. `st.session_state.weights`, nie `_weights`)
- **Klucze widgetów Streamlit:** prefiks zwykle wskazuje moduł (`"gen_matrix"`, `"log_run"`, `"hc_run"`, `"wc_generate"`)
- **Klucze wykresów Plotly:** prefiks `pc_` (plotly chart), np. `"pc_ols_VAR_resid"`. PAMIĘTAJ o uzupełnieniu klucza o `group_label` i indeks entry przy list-based wynikach — inaczej przy splicie Streamlit rzuci `StreamlitDuplicateElementKey`

## Wskaźniki w UI
- 🔀 = podział na podzbiory (split)
- ⚖️ = ważenie
- Żółty box (#FFF4CE + #D97706) = split
- Zielony box (#E2F0D9 + #548235) = wagi

---

## Najczęstsze błędy z którymi już się spotkaliśmy

1. **StreamlitDuplicateElementKey** — dodaj `group_label` i indeks do klucza wykresu
2. **f-string backslash error (Python 3.10)** — wyciągnij escape do zmiennej
3. **Chi² "zero element"** — puste wiersze/kolumny w krzyżówce, sprawdź `row_sums` i `col_sums` przed `stats.chi2_contingency(obs)`
4. **Unicode w pliku** — po każdej większej edycji przepuść plik przez skrypt konwersji na ASCII

---

## Typowe polecenia użytkownika i jak je interpretować

- **"uruchom aplikację"** → `streamlit run generator.py`
- **"pokaż błąd"** → użytkownik wklei traceback, naprawiaj bez zbędnego pytania o potwierdzenie
- **"dodaj funkcję X"** → rozszerz istniejący moduł, trzymaj się konwencji (`_merge_result`, split support, weights support, badge w expanderze, del button)
- **"napraw błąd Y"** → znajdź źródło, napraw minimalną zmianą, zweryfikuj składnię ASCII

---

## Po każdej istotnej zmianie

1. Sprawdź składnię: `python -m py_compile generator.py`
2. Sprawdź ASCII: spróbuj `open('generator.py','rb').read().decode('ascii')`
3. Uruchom aplikację: `streamlit run generator.py` i popatrz czy działa
4. Jeśli jest git, zrób commit z czytelnym message po polsku (np. `fix: poprawa obsługi pustych komórek w chi-kwadrat`)

---

## Czego NIE rób bez wyraźnej prośby

- NIE rozbijaj UI (`elif menu`) z `generator.py` na osobne pliki bez wyraźnego polecenia (warstwy `sao_core.py`/`sao_admin.py` są już wydzielone — to jest docelowa struktura)
- NIE zmieniaj UI z polskiego na angielski
- NIE wprowadzaj nowych dużych dependencji bez pytania (requirements.txt jest zachowawczy)
- NIE zmieniaj istniejących konwencji bez dobrego powodu (np. sposobu kumulowania wyników, kolorów wskaźników)
- NIE używaj bezpośrednich znaków Unicode w kodzie — zawsze escape `\uXXXX`
