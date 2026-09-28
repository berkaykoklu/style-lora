# Değerlendirme formu

48 kart, üç soru, tek sayfa. Görseller sayfanın içinde gömülü; değerlendiren
kişinin hiçbir şey kurması, üye olması ya da dosya yollaması gerekmiyor.

## Deploy — tamamen tarayıcıdan

Yerel makinede hiçbir şey kurmuyorsun. `npm install` gerekmiyor: bağımlılıkları
Vercel kendi tarafında kuruyor.

1. **vercel.com/new** → `berkaykoklu/style-lora` reposunu içe aktar
2. **Root Directory** → `web` seç (önemli, yoksa Python reposunu deploy eder)
3. **Deploy**
4. Proje açılınca **Storage** → **Create Database** → **Blob** → projeye bağla
   (bağlanınca `BLOB_READ_WRITE_TOKEN` otomatik geliyor)
5. **Settings → Environment Variables** → `RESULTS_TOKEN` = kendi seçeceğin parola
6. **Deployments** → en üstteki → **Redeploy** (4 ve 5'teki değişkenlerin gelmesi için)

Telefondan da yapılabilir.

## Sonuçlar

```
https://<proje>.vercel.app/api/results?token=<RESULTS_TOKEN>
```

Her cevap `responses/` altında ayrı bir JSON. Token'sız 401.

## Cevap anahtarı

`../key.json` hangi kartın hangi adaptörden geldiğini tutuyor ve **bu klasörde
değil** — `public/` altındaki her şey yayınlanıyor, anahtar da yayınlansaydı
değerlendirme körleme olmaktan çıkardı. Repoya da girmiyor.

Anahtarı kaybetme: onsuz cevaplar puanlanamaz.

## Yerelde denemek istersen (gerekli değil)

```bash
cd web && npm install && npx vercel dev
```
