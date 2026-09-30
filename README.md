# Yerel RAG Asistanı (Foundry Local)

Kendi bilgisayarında, tamamen offline çalışan, dokümanlarına bakarak
soru cevaplayan bir yapay zeka asistanı. Microsoft **Foundry Local**
ile yerel (on-device) model çalıştırır, **RAG** (Retrieval-Augmented
Generation) desenini kullanır, parçaları ve embedding'lerini
**SQLite**'ta saklar.

## Mimari

1. `documents/` klasöründeki `.txt` ve `.docx` dosyaları paragraflara bölünür (chunk).
2. Her parça embedding modeliyle bir vektöre dönüştürülür, SQLite'a kaydedilir (`ingest.py`).
3. Soru sorduğunda, soru da embedding'lenir; kosinüs benzerliğiyle en alakalı parçalar bulunur.
4. Bulunan parçalar, yerel sohbet modeline bağlam olarak verilir; model sadece bu bağlama dayanarak cevap üretir.

## Kullanılan modeller

Bu projede macOS (Apple Silicon) üzerinde bazı modellerin GPU
varyantlarında sayısal kararsızlık (NaN/Infinity) hatası tespit
edildi. Bu yüzden hem embedding hem sohbet modeli **CPU** varyantı
olarak sabitlendi:

- Embedding: `qwen3-embedding-0.6b-generic-cpu:1`
- Sohbet: `qwen2.5-1.5b-instruct-generic-cpu:4`

## 1) Ön koşullar

- macOS (Apple Silicon veya Intel)
- Python 3.11+
- Homebrew

## 2) Foundry Local kurulumu

```bash
brew tap microsoft/foundrylocal
brew install foundrylocal
brew trust microsoft/foundrylocal   # gerekirse
```

Servisin çalıştığını doğrula:

```bash
foundry service status
```

"Model management service is not running" yazarsa:

```bash
foundry service start
```

## 3) Proje ortamı

```bash
cd local-rag-project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4) Dokümanlarını ekle

`documents/` klasörüne `.txt` veya `.docx` dosyalarını koy. Örnekler
zaten mevcut (Foundry Local, RAG kavramı, SQLite, embedding hakkında)
— istersen sil, istersen bırak.

## 5) Dokümanları işle

```bash
python ingest.py
```

İlk çalıştırmada embedding modelini indirir. `knowledge_base.db`
dosyası oluşur.

Yeni doküman ekledikçe veya mevcutları değiştirdikçe bu adımı tekrarla.

## 6) Çalıştır

**CLI:**
```bash
python main.py
```

**Web arayüzü (Streamlit):**
```bash
streamlit run app.py
```

Tarayıcıda `http://localhost:8501` (veya benzeri bir port) açılır.
Sohbet kutusuna soru yazıp Enter'a basman yeterli. Kaynaklar cevabın
altında gösterilir.

## Bilinen sınırlamalar (küçük model dengesi)

Bu projede sohbet modeli olarak küçük (1.5B parametre) bir yerel
model kullanılıyor. Sistematik testler (`test.py`) sırasında şu
denge gözlemlendi:

- **Katı sistem promptu** (örnek/kesin kurallarla): Model, alakasız
  soruları doğru reddediyor ama bazen belgede GERÇEKTEN olan bilgiyi
  de "belgede yok" diyerek yanlışlıkla reddedebiliyor (yanlış negatif).
- **Esnek sistem promptu** (mevcut, `main.py`/`app.py`/`test.py`'de
  kullanılan): Model belgedeki bilgiyi daha iyi buluyor ve kullanıyor,
  ama bazen belgeyle hiçbir ilgisi olmayan sorularda (örn. "Fransa'nın
  başkenti neresi?") kendi genel bilgisinden halüsinasyon yapabiliyor.

10 soruluk test setinde (7 cevaplanabilir + 3 cevaplanamaz) tipik
sonuç: ~7/10 doğru, ~2/10 halüsinasyon, ~1/10 yanlış negatif. Bu,
1.5B boyutundaki bir modelin doğal kapasite sınırıdır; daha büyük
bir model (3B+) ile bu oran iyileşir ama hız/RAM kullanımı artar.

Retrieval (doğru parçayı bulma) katmanı bu sınırlamadan etkilenmez —
testlerle doğrulandığı üzere SQLite + kosinüs benzerliği katmanı
tutarlı şekilde doğru parçaları buluyor. Sorun sadece generation
(modelin bulunan bilgiyi doğru yorumlaması) katmanında.

## Sorun giderme

- **Embedding modeli "Infinity/NaN" hatası veriyor** → GPU varyantı
  yerine CPU varyantını kullan (`get_model_variant` ile tam model ID'si
  vererek, örn. `qwen3-embedding-0.6b-generic-cpu:1`).
- **Model tutarsız/tekrarlayan cevaplar veriyor** → Küçük modellerde
  (0.5B-1.5B) beklenen bir durum. `main.py`/`app.py` içinde tekrar
  koruma mantığı (`full_response` uzunluk/tekrar kontrolü) zaten var.
- **Streamlit `GZipResponder` hatası veriyor** → `starlette` sürümü
  uyumsuz, `pip install "starlette<0.47" --force-reinstall` çalıştır.
- **Cevaplar alakasız** → `documents/` içeriğini kontrol et, `ingest.py`
  tekrar çalıştır. `main.py`/`app.py` içindeki `TOP_K` değerini
  artırmayı dene.

## Dosya yapısı

```
local-rag-project/
  README.md
  requirements.txt
  db.py              - SQLite yardımcı fonksiyonları
  ingest.py          - Dokümanları chunk'la, embed'le, DB'ye kaydet
  main.py            - CLI soru-cevap döngüsü
  app.py             - Streamlit web arayüzü
  test.py            - Otomatik test seti
  documents/         - .txt / .docx dokümanların
  knowledge_base.db  - SQLite veritabanı (ingest.py ile oluşur)
```
