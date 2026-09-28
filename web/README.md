# Değerlendirme formu

48 kart, üç soru, tek sayfa. Görseller sayfanın içinde; değerlendiren kişinin
hiçbir şey kurması, üye olması ya da dosya yollaması gerekmiyor.

## Kurulum

```bash
cd web
npm install
vercel                       # projeyi bağla
vercel blob create-store style-eval
vercel env add RESULTS_TOKEN # kendi seçeceğin bir parola
vercel --prod
```

Blob deposu projeye bağlandığında `BLOB_READ_WRITE_TOKEN` otomatik geliyor.

## Sonuçlar

```
https://<proje>.vercel.app/api/results?token=<RESULTS_TOKEN>
```

Her cevap `responses/` altında ayrı bir JSON. Endpoint token olmadan 401 veriyor.

## Cevap anahtarı

`../key.json` hangi kartın hangi adaptörden geldiğini tutuyor ve **bu klasörde
değil** — `public/` altındaki her şey yayınlanıyor, anahtar da yayınlansaydı
değerlendirme körleme olmaktan çıkardı.

Anahtarı kaybetme: onsuz cevaplar puanlanamaz.
