# Lyrics Topic Modeling — Findings

**English** | [Türkçe](#türkçe)

> Every number below comes from the computed tables and figures under `v2/results/`. No lyrics are included (copyright); only aggregate counts and topic keywords.
>
> **v1 (2022) as historical reference:** `notebooks/` and `results/` at the repo root belong to the original hand-written 2022 study and are frozen. The old v1 CSVs no longer exist, so v1 was not re-run; it is referenced only for its methodology (lines joined without spaces, a hand-fixed `num_topics=30`, no quantitative metric). The 2016–2022 US period was rebuilt from Billboard Hot 100 + LRCLIB with the same v2 pipeline.
>
> **Slices:** `US_2016_2022` (565 songs), `US` 2023–2025 (369), `GB` 2023–2025 (400), and the v3 combined corpus `ALL` 2016–2025 (1,070 unique songs after deduplicating 1,334 records).
>
> **Per-topic visuals:** one v1-style banner word cloud per topic in [`wordclouds/`](wordclouds/): one folder per slice and model, top 5 topics by song count. In the 2023–2025 slices, the Korean word "아파트" from Rosé & Bruno Mars' *APT.* is dominant enough to become a topic keyword in more than one model.

---

## Finding 1 — Model generations (2003 → 2024): FASTopic leads

* **Observation:** On the same cleaned data, **FASTopic (2024)** reached topic diversity `1.0000` (zero word overlap) in all three period slices, and the highest coherence on `US_2016_2022` (`c_v = 0.4862`) and `GB` (`0.4500`). On `US` 2023–2025 it came second (`0.4159`) behind Top2Vec (`0.4224`), which found only 2 topics. Classic **LDA** had the lowest diversity (`0.5000–0.5467`) and low `c_v` (`0.3529–0.4003`).
* **Evidence:** [`compare/all_metrics_summary.csv`](compare/all_metrics_summary.csv), [`compare/model_comparison.png`](compare/model_comparison.png).

| Model | `US_2016_2022` `c_v` (Div / Topics) | `US 2023–2025` `c_v` (Div / Topics) | `GB 2023–2025` `c_v` (Div / Topics) |
|---|---|---|---|
| `lda` (2003) | 0.4003 (0.5467 / 15) | 0.3560 (0.5000 / 15) | 0.3529 (0.5667 / 15) |
| `nmf` | 0.4234 (0.6667 / 15) | 0.3880 (0.6267 / 15) | 0.4314 (0.6600 / 15) |
| `top2vec` | 0.4759 (1.0000 / 2) | **0.4224** (1.0000 / 2) | 0.3521 (1.0000 / 5) |
| `bertopic` | 0.3728 (0.7333 / 9) | 0.3736 (0.7143 / 7) | 0.3607 (0.7333 / 6) |
| `fastopic` (2024) | **0.4862** (**1.0000** / 15) | 0.4159 (**1.0000** / 15) | **0.4500** (**1.0000** / 15) |

* **Possible explanation:** FASTopic's optimal-transport alignment of documents, topics and words, on top of pretrained `all-MiniLM-L6-v2` embeddings, prevents word overlap between topics in short, repetitive texts like lyrics. Top2Vec collapses to 2–5 coarse clusters on corpora of 369–565 documents.
* **Limitation:** `c_v` is a word co-occurrence statistic and does not guarantee semantic quality on its own. FASTopic ran with a fixed `k=15`.

---

## Finding 2 — v1's fixed `k=30` vs a `k=5..40` coherence scan

* **Observation:** v1 fixed `num_topics=30` for LDA and NMF without justification. In the v2 `k = 5..40` scan, LDA peaks at **`k=10`** on `US_2016_2022` (`c_v = 0.4089`, diversity `0.5300`) and drops to `0.3622` at `k=30`. On the smaller 2023–2025 slices LDA peaks at **`k=5`** (`US: 0.3558`, `GB: 0.3801`).
* **Evidence:** [`eval/k_scan_US_2016_2022.csv`](eval/k_scan_US_2016_2022.csv), [`eval/k_scan_US.csv`](eval/k_scan_US.csv), [`eval/k_scan_GB.csv`](eval/k_scan_GB.csv), `eval/k_scan_*.png`.
* **Possible explanation:** With ~369–565 songs, 30 topics over-fragment the corpus; topics with few documents fill up with generic repeated words (LDA diversity `0.46–0.49` at `k=30`).
* **Limitation:** NMF's `c_v` keeps rising at high `k` (`k=35–40`: `0.4375–0.4668`) while diversity falls from `0.72–0.78` to `0.545–0.609`; choosing `k` from a single metric can mislead.

---

## Finding 3 — BERTopic `reduce_outliers` and the small-corpus limit

* **Observation:** With default HDBSCAN clustering, BERTopic put 35–42% of documents in the outlier cluster `-1` (`US_2016_2022`: 237/565, `US`: 157/369, `GB`: 141/400). After `reduce_outliers(strategy="c-tf-idf")` outliers dropped to **0** in all slices, at the cost of coherence: `US` `0.4052 → 0.3736`, `GB` `0.3765 → 0.3607` (`US_2016_2022`: `0.3691 → 0.3728`).
* **Evidence:** first run before outlier reduction vs [`compare/all_metrics_summary.csv`](compare/all_metrics_summary.csv); [`models/topics_US_bertopic.csv`](models/topics_US_bertopic.csv).
* **Possible explanation:** Lyrics often mix several themes (love + party + slang), so density-based HDBSCAN leaves borderline songs as outliers; forcing them into the nearest c-TF-IDF cluster gives 100% coverage but slightly dilutes topic purity.
* **Limitation:** `min_topic_size=8` was tuned for 369–565-document slices; larger corpora may show fewer outliers.

---

## Finding 4 — 2016–2022 vs 2023–2025: a period shift and a genre confounder

* **Observation:** Comparing US `2016–2022` with `2023–2025`, introspective and relationship words rise (`think +2.947`, `good +2.055`, `leave +1.946`, `fall +1.785`, `find +1.747`, `love +1.167` per 1,000 tokens) while hip-hop slang and materialist terms fall (`get -6.367`, `low -3.673`, `nigga -3.122`, `gon -2.869`, `money -2.249`, `bitch -1.325`, `gang -1.241`). The lexical genre proxy shows `hiphop_proxy_share` at 50.6–51.2% in 2017–2018, 30.5% in 2024 and **25.8%** in 2025, while `romance_proxy_share` stays in the 63–73% band.
* **Evidence:** [`compare/period_comparison_US.csv`](compare/period_comparison_US.csv), [`compare/genre_proxy_summary.csv`](compare/genre_proxy_summary.csv), [`compare/topics_over_time_US_2016_2025.png`](compare/topics_over_time_US_2016_2025.png), [`compare/wordcloud_US_2016_2022.png`](compare/wordcloud_US_2016_2022.png), [`compare/wordcloud_US_2023_2025.png`](compare/wordcloud_US_2023_2025.png).
* **Possible explanation:** This does not on its own mean "people's feelings changed". The trap/hip-hop weight of the 2017–2019 US charts shrinking in favor of pop, country and indie-pop in 2023–2025 (genre composition) can explain most of the word shift.
* **Limitations:**
  1. **Sampling difference:** `US_2016_2022` comes from Billboard Hot 100 year-end charts, `US` 2023–2025 from Spotify search ranking; the two sources do not measure popularity the same way.
  2. **No genre labels:** the data has no genre column, so the genre effect is measured with a keyword proxy (`HIPHOP_SLANG_PROXY` / `ROMANCE_POP_PROXY`), not real labels.

---

## Finding 5 — US vs UK (2023–2025)

* **Observation:** In 2023–2025, the **GB** corpus uses emotional/relationship words more often than **US** (`want +2.657`, `love +2.645`, `feel +2.625`, `night +1.346`, `away +1.179` per 1,000 tokens). BERTopic agrees: GB's largest cluster is `love / hard / hold` (145/400 songs, 36.3%), US's is `bitch / fuck / shit` (96/369, 26.0%).
* **Evidence:** [`compare/country_comparison_2023_2025.csv`](compare/country_comparison_2023_2025.csv), [`models/topics_GB_bertopic.csv`](models/topics_GB_bertopic.csv), [`models/topics_US_bertopic.csv`](models/topics_US_bertopic.csv), [`compare/wordcloud_GB_2023_2025.png`](compare/wordcloud_GB_2023_2025.png).
* **Possible explanation:** UK charts appear to carry more dance-pop and ballads, US charts more hip-hop (`genre_proxy_summary.csv`, 2023–2025 average `hiphop_proxy_share`: US 33.1% vs GB 28.3%).
* **Limitation:** Spotify `search` relevance ranking per market is not an official sales or streaming chart.

---

## Finding 6 — v3: the combined decade (2016–2025, US + UK)

* **Observation:**
  1. **Words of the decade:** across 1,070 songs the top three words are `know` (3,548), `get` (3,150) and `love` (2,470). `love` is in every year's top 5 and rises to #2 in 2024 and 2025. `bitch` (2017, 2019) and `nigga` (2019) reach the yearly top 10 only before 2020; after that the only explicit top-10 entry is `shit` at #10 in 2021. `think` enters the top 10 in 2023–2024 and `leave` in 2025.
  2. **Model collapse on a heterogeneous corpus:** on `ALL`, Top2Vec and BERTopic collapse to **2 topics** each. LDA, NMF and FASTopic keep 15; only FASTopic does so with zero word overlap.

  | Model | Topics | `c_v` | Diversity |
  |---|---|---|---|
  | `lda` | 15 | 0.3846 | 0.5533 |
  | `nmf` | 15 | 0.3960 | 0.6533 |
  | `top2vec` | 2 | 0.5271 | 1.0000 |
  | `bertopic` | 2 | 0.4365 | 0.8000 |
  | `fastopic` | 15 | **0.4449** | **1.0000** |

  3. **The decade's largest topic is mainly American:** FASTopic's biggest topic, `bitch / money / nigga` (154 songs, 14%), comes 121 songs from US-only charts, 9 from UK-only and 24 from both. Softer topics have a clearly larger UK and shared share (`midnight / feather / sunrise`: 41 US, 12 GB, 25 both). Nearly half of the explicit topic (48%) falls in just four years, 2016–2019: 18.5 songs a year on average versus 13.3 in 2020–2025.
  4. **Market overlap:** of the 1,070 songs, 670 appear only in US lists, 139 only in UK lists and 261 in both.
* **Evidence:** [`eval/metrics_ALL.csv`](eval/metrics_ALL.csv), [`eval/k_scan_ALL.csv`](eval/k_scan_ALL.csv), [`models/topics_ALL_fastopic.csv`](models/topics_ALL_fastopic.csv), [`compare/decade_top_words_ALL.csv`](compare/decade_top_words_ALL.csv), [`compare/yearly_top10_words_ALL.csv`](compare/yearly_top10_words_ALL.csv), [`compare/markets_ALL.csv`](compare/markets_ALL.csv), [`compare/topics_by_market_ALL_fastopic.csv`](compare/topics_by_market_ALL_fastopic.csv), [`wordclouds/ALL_fastopic/`](wordclouds/ALL_fastopic/).
* **Possible explanation:** Top2Vec and BERTopic cluster in embedding space; on a mixed corpus the strongest split ("explicit hip-hop" vs "everything else") dominates and smaller themes merge. Their high `c_v` is an artifact: keeping 2 topics coherent is easy, but 2 clusters do not describe 1,070 songs. FASTopic's transport plan spreads documents over a fixed number of topics, so it keeps fine-grained themes.
* **Limitations:** UK data covers only 2023–2025, so "US-only" is partly a time effect: every pre-2023 song is US-only by construction. Deduplication keeps each song's earliest year. LDA's k-scan on `ALL` peaks at `k=5` (`0.4044`), so `k=15` is a readability choice, not a coherence optimum.

---

<a name="türkçe"></a>

# Şarkı Sözlerinde Konu Modelleme — Bulgular

[English](#lyrics-topic-modeling--findings) | **Türkçe**

> Aşağıdaki tüm sayılar `v2/results/` altındaki hesaplanmış tablo ve grafiklerden gelir. Şarkı sözü yoktur (telif); yalnızca toplu sayımlar ve konu kelimeleri vardır.
>
> **Tarihî referans (v1, 2022):** Repo kökündeki `notebooks/` ve `results/` klasörleri 2022'de el yazımı yapılan ilk çalışmaya aittir ve dondurulmuştur. Eski v1 CSV'leri artık mevcut olmadığından v1 yeniden çalıştırılmamış, yalnızca metodolojisiyle (satırların boşluksuz birleştirilmesi, elle sabitlenen `num_topics=30`, nicel metrik yokluğu) referans alınmıştır. 2016–2022 ABD dönemi Billboard Hot 100 + LRCLIB üzerinden aynı v2 hattıyla yeniden üretilmiştir.
>
> **Dilimler:** `US_2016_2022` (565 şarkı), `US` 2023–2025 (369), `GB` 2023–2025 (400) ve v3 birleşik korpusu `ALL` 2016–2025 (1.334 kayıt tekilleştirilince 1.070 şarkı).
>
> **Konu bazlı görseller:** Her konu için v1 stilinde tek banner kelime bulutu [`wordclouds/`](wordclouds/) altındadır: dilim ve model başına bir klasör, her klasörde şarkı sayısına göre ilk 5 konu. 2023–2025 dilimlerinde Rosé & Bruno Mars'ın *APT.* şarkısındaki Korece "아파트" kelimesi birden fazla modelde konu anahtar kelimesi olacak kadar baskındır.

---

## Bulgu 1 — Model kuşakları (2003 → 2024): FASTopic önde

* **Gözlem:** Aynı temizlenmiş veride **FASTopic (2024)** üç dönem diliminin hepsinde `1.0000` topic diversity (sıfır kelime tekrarı) elde etti; `US_2016_2022` (`c_v = 0.4862`) ve `GB` (`0.4500`) dilimlerinde en yüksek coherence skoruna ulaştı. `US` 2023–2025'te `0.4159` ile, yalnızca 2 konu bulan Top2Vec'in (`0.4224`) ardından ikinci oldu. Klasik **LDA** en düşük diversity (`0.5000–0.5467`) ve düşük `c_v` (`0.3529–0.4003`) üretti.
* **Kanıt:** [`compare/all_metrics_summary.csv`](compare/all_metrics_summary.csv), [`compare/model_comparison.png`](compare/model_comparison.png). Tablo İngilizce bölümdedir.
* **Olası açıklama:** FASTopic'in optimal transport tabanlı belge-konu-kelime hizalaması ve önceden eğitilmiş `all-MiniLM-L6-v2` gömüleri, şarkı sözü gibi kısa ve tekrarlı metinlerde konular arası kelime çakışmasını engelliyor. Top2Vec 369–565 belgelik korpuslarda 2–5 kaba kümeye çöküyor.
* **Sınırlama:** `c_v` kelime birlikteliği istatistiğidir, anlamsal kaliteyi tek başına garanti etmez. FASTopic sabit `k=15` ile koştu.

---

## Bulgu 2 — v1'in sabit `k=30` seçimi ve `k=5..40` taraması

* **Gözlem:** v1'de LDA ve NMF için konu sayısı gerekçesiz biçimde `num_topics=30` sabitlenmişti. v2'nin `k = 5..40` taramasında LDA, `US_2016_2022`'de **`k=10`**'da tepe yapıyor (`c_v = 0.4089`, diversity `0.5300`) ve `k=30`'da `0.3622`'ye düşüyor. Daha küçük 2023–2025 dilimlerinde LDA **`k=5`**'te zirvede (`US: 0.3558`, `GB: 0.3801`).
* **Kanıt:** [`eval/k_scan_US_2016_2022.csv`](eval/k_scan_US_2016_2022.csv), [`eval/k_scan_US.csv`](eval/k_scan_US.csv), [`eval/k_scan_GB.csv`](eval/k_scan_GB.csv), `eval/k_scan_*.png`.
* **Olası açıklama:** ~369–565 şarkılık korpusta 30 konu aşırı parçalanmaya yol açıyor; az belgeye düşen konular genel ve tekrarlı kelimelerle doluyor (`k=30`'da LDA diversity `0.46–0.49`).
* **Sınırlama:** NMF'te `c_v` yüksek `k`'da artmaya devam ediyor (`k=35–40`: `0.4375–0.4668`), ama diversity `0.72–0.78`'den `0.545–0.609`'a düşüyor; `k`'yı tek metrikle seçmek yanıltabilir.

---

## Bulgu 3 — BERTopic `reduce_outliers` etkisi ve küçük korpus sınırı

* **Gözlem:** Varsayılan HDBSCAN kümelemesiyle BERTopic belgelerin %35–42'sini aykırı (`-1`) kümesine attı (`US_2016_2022`: 237/565, `US`: 157/369, `GB`: 141/400). `reduce_outliers(strategy="c-tf-idf")` sonrası aykırı sayısı her dilimde **0**'a indi; karşılığında coherence düştü: `US` `0.4052 → 0.3736`, `GB` `0.3765 → 0.3607` (`US_2016_2022`: `0.3691 → 0.3728`).
* **Kanıt:** aykırı azaltma öncesi ilk koşu ile [`compare/all_metrics_summary.csv`](compare/all_metrics_summary.csv) kıyası; [`models/topics_US_bertopic.csv`](models/topics_US_bertopic.csv).
* **Olası açıklama:** Şarkı sözleri sık sık birden fazla temayı karıştırıyor (aşk + parti + argo); yoğunluk tabanlı HDBSCAN sınırdaki şarkıları aykırı sayıyor. Onları en yakın c-TF-IDF kümesine zorla atamak kapsamı %100'e çıkarıyor ama konu saflığını hafifçe seyreltiyor.
* **Sınırlama:** `min_topic_size=8`, 369–565 belgelik dilimlere göre seçildi; daha büyük korpuslarda aykırı oranı düşebilir.

---

## Bulgu 4 — 2016–2022 ile 2023–2025: dönem kayması ve tür karıştırıcısı

* **Gözlem:** ABD'de `2016–2022` ile `2023–2025` kıyaslandığında içe dönük ve ilişki kelimeleri yükseliyor (`think +2.947`, `good +2.055`, `leave +1.946`, `fall +1.785`, `find +1.747`, `love +1.167` / 1.000 token); hip-hop argosu ve materyalist terimler geriliyor (`get -6.367`, `low -3.673`, `nigga -3.122`, `gon -2.869`, `money -2.249`, `bitch -1.325`, `gang -1.241`). Sözlüksel tür vekilinde `hiphop_proxy_share` 2017–2018'de %50,6–51,2 iken 2024'te %30,5'e, 2025'te **%25,8**'e düşüyor; `romance_proxy_share` %63–73 bandında kalıyor.
* **Kanıt:** [`compare/period_comparison_US.csv`](compare/period_comparison_US.csv), [`compare/genre_proxy_summary.csv`](compare/genre_proxy_summary.csv), [`compare/topics_over_time_US_2016_2025.png`](compare/topics_over_time_US_2016_2025.png), [`compare/wordcloud_US_2016_2022.png`](compare/wordcloud_US_2016_2022.png), [`compare/wordcloud_US_2023_2025.png`](compare/wordcloud_US_2023_2025.png).
* **Olası açıklama:** Bu tek başına "insanların duyguları değişti" demek değildir. 2017–2019 ABD listelerindeki trap/hip-hop ağırlığının 2023–2025'te pop, country ve indie-pop lehine azalması (tür bileşimi) kelime kaymasının büyük kısmını açıklayabilir.
* **Sınırlamalar:**
  1. **Örnekleme farkı:** `US_2016_2022` Billboard Hot 100 yıl sonu listelerinden, `US` 2023–2025 Spotify arama sıralamasından geliyor; iki kaynak popülerliği aynı şekilde ölçmüyor.
  2. **Tür etiketi yok:** Veride tür kolonu bulunmadığı için tür etkisi gerçek etiketlerle değil, anahtar kelime vekiliyle (`HIPHOP_SLANG_PROXY` / `ROMANCE_POP_PROXY`) ölçüldü.

---

## Bulgu 5 — ABD ve Birleşik Krallık (2023–2025)

* **Gözlem:** 2023–2025'te **GB** korpusunda duygusal ve ilişki kelimeleri **US**'e göre daha sık geçiyor (`want +2.657`, `love +2.645`, `feel +2.625`, `night +1.346`, `away +1.179` / 1.000 token). BERTopic de aynı yönde: GB'nin en büyük kümesi `love / hard / hold` (145/400 şarkı, %36,3), US'inki `bitch / fuck / shit` (96/369, %26,0).
* **Kanıt:** [`compare/country_comparison_2023_2025.csv`](compare/country_comparison_2023_2025.csv), [`models/topics_GB_bertopic.csv`](models/topics_GB_bertopic.csv), [`models/topics_US_bertopic.csv`](models/topics_US_bertopic.csv), [`compare/wordcloud_GB_2023_2025.png`](compare/wordcloud_GB_2023_2025.png).
* **Olası açıklama:** GB listelerinde dans-pop ve baladların, US listelerinde hip-hop'un payı görece yüksek görünüyor (`genre_proxy_summary.csv`, 2023–2025 ortalama `hiphop_proxy_share`: US %33,1, GB %28,3).
* **Sınırlama:** Spotify `search` endpoint'inin ülke bazlı alaka sıralaması resmî satış ya da dinlenme listesi değildir.

---

## Bulgu 6 — v3: birleşik on yıl (2016–2025, ABD + Birleşik Krallık)

* **Gözlem:**
  1. **On yılın kelimeleri:** 1.070 şarkıda ilk üç kelime `know` (3.548), `get` (3.150) ve `love` (2.470). `love` her yılın ilk 5'inde yer alıyor, 2024 ve 2025'te 2. sıraya çıkıyor. `bitch` (2017, 2019) ve `nigga` (2019) yıllık ilk 10'a yalnızca 2020'den önce giriyor; sonrasında ilk 10'daki tek müstehcen kelime 2021'de 10. sıradaki `shit`. `think` 2023–2024'te, `leave` 2025'te ilk 10'a giriyor.
  2. **Heterojen korpusta model çöküşü:** `ALL` diliminde Top2Vec ve BERTopic **2 konuya** çöküyor. LDA, NMF ve FASTopic 15 konuyu koruyor; bunu sıfır kelime tekrarıyla yapan tek model FASTopic (`c_v 0.4449`, diversity `1.0000`). Tablo İngilizce bölümdedir.
  3. **On yılın en büyük konusu ağırlıkla Amerikan:** FASTopic'in en büyük konusu `bitch / money / nigga` (154 şarkı, %14); bunların 121'i yalnızca ABD listelerinden, 9'u yalnızca Birleşik Krallık listelerinden, 24'ü ikisinden birden geliyor. Yumuşak konularda Birleşik Krallık ve ortak hitlerin payı belirgin şekilde yüksek (`midnight / feather / sunrise`: 41 US, 12 GB, 25 ortak). Müstehcen konunun yarısına yakını (%48) yalnızca dört yıla, 2016–2019'a düşüyor: yılda ortalama 18,5 şarkı, 2020–2025'te ise 13,3.
  4. **Pazar örtüşmesi:** 1.070 şarkının 670'i yalnızca ABD listelerinde, 139'u yalnızca Birleşik Krallık listelerinde, 261'i ikisinde birden yer alıyor.
* **Kanıt:** [`eval/metrics_ALL.csv`](eval/metrics_ALL.csv), [`eval/k_scan_ALL.csv`](eval/k_scan_ALL.csv), [`models/topics_ALL_fastopic.csv`](models/topics_ALL_fastopic.csv), [`compare/decade_top_words_ALL.csv`](compare/decade_top_words_ALL.csv), [`compare/yearly_top10_words_ALL.csv`](compare/yearly_top10_words_ALL.csv), [`compare/markets_ALL.csv`](compare/markets_ALL.csv), [`compare/topics_by_market_ALL_fastopic.csv`](compare/topics_by_market_ALL_fastopic.csv), [`wordclouds/ALL_fastopic/`](wordclouds/ALL_fastopic/).
* **Olası açıklama:** Top2Vec ve BERTopic gömü uzayında kümeleme yapıyor; karışık bir korpusta en güçlü ayrım ("müstehcen hip-hop" ve "geri kalan her şey") baskın çıkıyor, küçük temalar birleşiyor. Yüksek `c_v` skorları bu yüzden yanıltıcı: 2 konuyu tutarlı tutmak kolaydır, ama 2 küme 1.070 şarkıyı anlatmaz. FASTopic'in transport planı belgeleri sabit sayıda konuya dağıttığı için ince temaları koruyor.
* **Sınırlamalar:** Birleşik Krallık verisi yalnızca 2023–2025'i kapsıyor; dolayısıyla "yalnızca ABD" kısmen bir zaman etkisi, çünkü 2023 öncesi her şarkı tanım gereği yalnızca ABD'de. Tekilleştirme her şarkının en erken yılını tutuyor. `ALL` üzerinde LDA'nın k taraması `k=5`'te (`0.4044`) tepe yapıyor; yani `k=15` coherence optimumu değil, okunabilirlik tercihi.
