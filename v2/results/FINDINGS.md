# Lyrics Topic Modeling v2 — Bulgular ve Analiz Özeti

> Medium yazısı ve portfolyo sunumu için ölçülmüş bulgu hammaddesi. Tüm sayısal değerler `v2/results/` altındaki hesaplanmış tablo ve grafiklerden alınmıştır.
> **Tarihî referans (v1, 2022):** Repo kökündeki `notebooks/` ve `results/` klasörleri 2022'de el yazımı yapılan ilk çalışmaya aittir ve dondurulmuştur. Eski v1 CSV'leri mevcut olmadığından modellemeye sokulmamış, yalnızca metodolojik bağlam (ör. satır birleştirme hatası, elle sabitlenen `num_topics=30`, nicel metrik yokluğu) olarak referans alınmıştır. Dönem kıyası için `2016–2022` ABD verisi Billboard Hot 100 + LRCLIB üzerinden aynı v2 hattıyla yeniden üretilmiştir.

> **Konu bazlı görseller:** Her konu için tek banner wordcloud (v1 stili) [`wordclouds/`](wordclouds/) altındadır; model/dilim başına bir klasör, her klasörde şarkı sayısına göre ilk 5 konu. Not: `2023–2025` dilimlerinde Rosé & Bruno Mars'ın *APT.* şarkısındaki Korece "아파트" kelimesi birden fazla modelde konu anahtar kelimesi olacak kadar baskındır.

---

## Bulgu 1 — Model Kuşağı Kıyası (2003 → 2024): FASTopic Üstünlüğü

* **Gözlem:** Beş modelin (`LDA → NMF → Top2Vec → BERTopic → FASTopic`) aynı temizlenmiş veri üzerinde ölçülen sonuçlarında **FASTopic (2024)** her üç dilimde de `1.0000` topic diversity (sıfır kelime tekrarı) elde etti; `US_2016_2022` (`c_v = 0.4862`) ve `GB` (`c_v = 0.4500`) dilimlerinde en yüksek coherence skoruna ulaştı, `US 2023–2025` diliminde ise `0.4159` ile `Top2Vec` (`0.4224`, ancak yalnızca 2 konu) hemen ardından ikinci oldu. Klasik **LDA** ise en düşük diversity (`0.5000–0.5467`) ve düşük `c_v` (`0.3529–0.4003`) üretti.
* **Kanıt:** [`compare/all_metrics_summary.csv`](compare/all_metrics_summary.csv) ve [`compare/model_comparison.png`](compare/model_comparison.png).

| Model | `US_2016_2022` `c_v` (Div / Konu) | `US 2023–2025` `c_v` (Div / Konu) | `GB 2023–2025` `c_v` (Div / Konu) |
|---|---|---|---|
| `lda` (2003) | 0.4003 (0.5467 / 15) | 0.3560 (0.5000 / 15) | 0.3529 (0.5667 / 15) |
| `nmf` | 0.4234 (0.6667 / 15) | 0.3880 (0.6267 / 15) | 0.4314 (0.6600 / 15) |
| `top2vec` | 0.4759 (1.0000 / 2) | **0.4224** (1.0000 / 2) | 0.3521 (1.0000 / 5) |
| `bertopic` | 0.3728 (0.7333 / 9) | 0.3736 (0.7143 / 7) | 0.3607 (0.7333 / 6) |
| `fastopic` (2024) | **0.4862** (**1.0000** / 15) | 0.4159 (**1.0000** / 15) | **0.4500** (**1.0000** / 15) |

* **Olası Açıklama:** FASTopic'in optimal transport tabanlı belge-konu-kelime hizalaması ve önceden eğitilmiş `all-MiniLM-L6-v2` gömüleri, şarkı sözü gibi tekrarlı ve kısa metinlerde konular arası kelime çakışmasını engelliyor; `Top2Vec` ise 369–565 belgelik görece küçük korpuslarda yalnızca 2–5 kaba kümeye çöküyor.
* **Belirsizlik / Sınırlama:** `c_v` coherence klasik kelime-birlikteliği istatistiğine dayanır; anlamsal tutarlılığı tek başına garanti etmez. Ayrıca FASTopic `k=15` sabit parametresiyle koştu.

---

## Bulgu 2 — v1'in Sabit `k=30` Seçimi vs `k=5..40` Coherence Taraması (`k=10` / `k=5` Bulgusu)

* **Gözlem:** v1 çalışmasında LDA ve NMF için konu sayısı gerekçesiz biçimde `num_topics=30` sabitlenmişti. v2'de yapılan `k = 5..40` taramasında LDA, `US_2016_2022` diliminde **`k=10`** noktasında tepe yaparken (`c_v = 0.4089`, diversity `0.5300`), `k=30`'da `0.3622`'ye düşüyor. Daha küçük `2023–2025` dilimlerinde ise LDA **`k=5`** noktasında zirve yapıyor (`US: 0.3558`, `GB: 0.3801`).
* **Kanıt:** [`eval/k_scan_US_2016_2022.csv`](eval/k_scan_US_2016_2022.csv), [`eval/k_scan_US.csv`](eval/k_scan_US.csv), [`eval/k_scan_GB.csv`](eval/k_scan_GB.csv) ve `eval/k_scan_*.png`.
* **Olası Açıklama:** ~369–565 şarkılık korpuslarda 30 konu aşırı parçalanmaya yol açıyor; az sayıda belgeye düşen konular tekrarlı genel kelimelerle doluyor (`k=30`'da LDA diversity `0.46–0.49`).
* **Belirsizlik / Sınırlama:** NMF'te `c_v` daha yüksek `k` değerlerinde (`k=35–40`: `0.4375–0.4668`) artmaya devam etse de diversity eşzamanlı olarak `0.72–0.78`'den `0.545–0.609`'a düşüyor; yani tek metrikle `k` seçmek yanıltıcı olabilir.

---

## Bulgu 3 — BERTopic `reduce_outliers` Etkisi ve Küçük Korpus Sınırlaması

* **Gözlem:** HDBSCAN varsayılan kümelemesiyle BERTopic belgelerin %35–42'sini aykırı (`-1`) kümesine attı (`US_2016_2022`: 237/565, `US`: 157/369, `GB`: 141/400). `reduce_outliers(strategy="c-tf-idf")` uygulandığında aykırı sayısı her üç dilimde de **0**'a indi; bunun karşılığında `US 2023–2025` `c_v` skoru `0.4052`'den `0.3736`'ya, `GB` `0.3765`'ten `0.3607`'ye geriledi (`US_2016_2022` ise `0.3691 → 0.3728`).
* **Kanıt:** `YP-004` (öncesi) ve `compare/all_metrics_summary.csv` (sonrası), [`models/topics_US_bertopic.csv`](models/topics_US_bertopic.csv).
* **Olası Açıklama:** Şarkı sözleri sıkça birden fazla temayı (aşk + parti + argo) karıştırdığı için yoğunluk tabanlı HDBSCAN sınırda kalan şarkıları `-1`'e atıyor; bunları en yakın c-TF-IDF kümesine zorla atamak kapsayıcılığı %100'e çıkarırken küme içi kelime saflığını hafifçe seyreltiyor.
* **Belirsizlik / Sınırlama:** `min_topic_size=8` seçimi ~369–565 belgelik dilimlere uyarlandı; daha büyük veri hacimlerinde HDBSCAN aykırı oranı daha düşük olabilir.

---

## Bulgu 4 — 2016–2022 vs 2023–2025 Dönem Kayması ve Tür Karıştırıcısı

* **Gözlem:** ABD listelerinde `2016–2022` (`US_2016_2022`) ile `2023–2025` (`US`) kıyaslandığında, içe dönük/ilişki kelimeleri belirgin biçimde yükselirken (`think +2.947`, `good +2.055`, `leave +1.946`, `fall +1.785`, `find +1.747`, `love +1.167` / 1.000 token), hip-hop/sokak jargonu ve materyalist terimler geriliyor (`get -6.367`, `low -3.673`, `nigga -3.122`, `gon -2.869`, `money -2.249`, `bitch -1.325`, `gang -1.241`). Sözlüksel tür vekili (`genre_proxy_summary.csv`) incelendiğinde, `hiphop_proxy_share` 2017–2018'de %50.6–%51.2 seviyesindeyken 2024'te %30.5'e, 2025'te **%25.8**'e düşüyor; `romance_proxy_share` ise %63–73 bandında kalıyor.
* **Kanıt:** [`compare/period_comparison_US.csv`](compare/period_comparison_US.csv), [`compare/genre_proxy_summary.csv`](compare/genre_proxy_summary.csv), [`compare/topics_over_time_US_2016_2025.png`](compare/topics_over_time_US_2016_2025.png), [`compare/wordcloud_US_2016_2022.png`](compare/wordcloud_US_2016_2022.png) ve [`compare/wordcloud_US_2023_2025.png`](compare/wordcloud_US_2023_2025.png).
* **Olası Açıklama:** Bu değişim tek başına "insanların duyguları değişti" demek değildir; 2017–2019 döneminde ABD Hot 100 listelerindeki trap/hip-hop ağırlığının 2023–2025'te pop, country ve indie-pop lehine azalması (tür bileşimi karıştırıcısı) bu kelime kaymasının büyük bölümünü açıklayabilir.
* **Belirsizlik / Sınırlama:**
  1. **Örnekleme farkı:** `US_2016_2022` Billboard Hot 100 yıl sonu listelerinden, `US 2023–2025` ise Spotify arama sıralamasından geldi; iki kaynak aynı popülerlik ölçütünü kullanmıyor (`MK-003`, `MK-008`).
  2. **Tür etiketi yokluğu:** Mevcut CSV'lerde `genre` sütunu boştur; bu nedenle tür etkisi gerçek tür etiketleriyle değil, anahtar kelime vekili (`HIPHOP_SLANG_PROXY` / `ROMANCE_POP_PROXY`) ile ölçülmüştür.

---

## Bulgu 5 — ABD vs Birleşik Krallık (2023–2025) Karşılaştırması

* **Gözlem:** 2023–2025 döneminde **GB** korpusunda duygusal/ilişki kelimeleri (`want +2.657`, `love +2.645`, `feel +2.625`, `night +1.346`, `away +1.179` / 1.000 token) **US**'e kıyasla daha yüksek sıklıkta geçiyor. BERTopic konularında da GB'nin en büyük kümesi `love / hard / hold` (145/400 şarkı, %36.3) iken, US'in en büyük kümesi `bitch / fuck / shit` (96/369 şarkı, %26.0) olarak ayrışıyor.
* **Kanıt:** [`compare/country_comparison_2023_2025.csv`](compare/country_comparison_2023_2025.csv), [`models/topics_GB_bertopic.csv`](models/topics_GB_bertopic.csv), [`models/topics_US_bertopic.csv`](models/topics_US_bertopic.csv) ve [`compare/wordcloud_GB_2023_2025.png`](compare/wordcloud_GB_2023_2025.png).
* **Olası Açıklama:** GB listelerinde dans-pop ve baladların payı, US listelerinde ise hip-hop/rap ağırlığı görece daha yüksek görünüyor (`genre_proxy_summary.csv`: 2023–2025 ortalama `hiphop_proxy_share` US %33.1 vs GB %28.3).
* **Belirsizlik / Sınırlama:** Spotify `search` endpoint'inin ülke bazlı alaka sıralaması gerçek resmi satış/dinlenme listesi değildir (`MK-003`).
