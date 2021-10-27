# Proyek Machine Learning - Recommender System 

## Table of contents

- [Project Overview](#project-overview)
- [Business Understanding](#business-understanding)
- [Data Understanding](#data-understanding)
- [Data Preparation](#data-preparation)
- [Modeling and Result](#modeling-and-result)
- [Evaluation](#evaluation)
- [References](#references)


## Project Overview
Proyek *machine learning* ini akan membahas tentang **"Sistem Rekomendasi Buku dengan Content Based Filtering dan Collaborative Filtering"**.

![banner](../main/images/banner.png "banner")

Membaca buku adalah salah satu kegiatan yang belum banyak peminatnya di Indonesia. Padahal membaca adalah jendela ilmu dan salah satu segmen pasar awal e-commerce Amazon pada awalnya adalah pasar buku sebelum akhirnya meluaskan target pasar ke segmen lainnya. Dalam e-commerce Amazon, tentunya terdapat sistem rekomendasi yang membantu pengunjung mencari buku agar pengunjung website tersebut membuat keputusan akhir yaitu membelinya [[1]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.23.9764&rep=rep1&type=pdf). Sistem rekomendasi buku juga dapat membantu pembaca buku mengetahui informasi buku-buku yang akan dibaca selanjutnya sesuai dengan preferensi pengguna dari rekam jejak buku yang telah dibaca sebelumnya. Terjadi keuntungan dua pihak karena konsumen mendapatkan informasi yang diperlukan untuk membuat suatu keputusan sementara dari pemilik bisnis juga mendapatkan profit dari penjualan. Hal ini menjadikan sistem rekomendasi telah digunakan secara luas oleh hampir semua area bisnis. [[2]](https://ejournal.akprind.ac.id/index.php/technoscientia/article/view/612).

Pada proyek machine learning ini, sistem rekomendasi yang akan dibuat memiliki fokus pada sistem rekomendasi buku karena di Indonesia sendiri masih diperlukan sebuah sistem yang dapat membantu merekomendasikan para pembaca agar lebih mudah mendapatkan informasi buku-buku yang akan dibaca selanjutnya. Hal ini didasarkan pada hasil survey kemampuan membaca oleh PISA tahun 2018 yang menghasilkan nilai skor membaca pelajar Indonesia yang masih sangat rendah [[3]](https://www.oecd.org/education/pisa-2018-results-volume-i-5f07c754-en.htm).

## Business Understanding
!!!
(Mengajukan dua atau lebih solution approach. Misalnya, menggunakan dua (atau lebih) algoritma sistem rekomendasi untuk mencapai solusi yang diinginkan.)
Bepergian adalah salah satu aktifitas mengunjungi tempat lain dengan tujuan tertentu. Mobilitas melewati udara atau penerbangan sudah menjadi hal yang biasa dan semakin  dijangkau masyarakat luas dalam beberapa dekade terakhir. Bepergian tidak lepas dari risiko yang dapat mengganggu kenyamanan selama perjalanan hingga kembali ke tempat asal. Risiko ini menjadi peluang bagi asuransi bepergian untuk menawarkan jasa perlindungan dari hal-hal yang tidak terduga seperti kemungkinan sakit, hilang passport, keterlambatan pesawat, atau risiko dengan rumah yang ditinggalkan. Perusahaan asuransi bepergian tentu memerlukan strategi untuk memasarkan jasanya dengan tepat sasaran sehingga biaya promosi atau *marketing* dapat menjadi lebih efisien. Untuk memenuhi keperluan tersebut, proyek ini akan membuat sebuah model *machine learning* dengan kemampuan klasifikasi pelanggan dalam pembelian asuransi.

### Problem Statements

Dari latar belakang di atas, dapat ditarik rumusan masalah sebagai berikut:
1. Bagaimana melakukan pra-pemrosesan data asuransi perjalanan agar menghasilkan data latih bagi *machine learning* prediksi keputusan pelanggan dalam pembelian asuransi perjalanan?
2. Bagaimana membuat model *machine learning* yang mampu memprediksi keputusan pelanggan dalam pembelian asuransi perjalanan?

### Goals
Tujuan proyek yang ingin dicapai adalah:
1. Melakukan pra-pemrosesan data asuransi perjalanan agar menghasilkan data latih yang cukup bagi *machine learning* prediksi keputusan pelanggan dalam pembelian asuransi perjalanan.
2. Membuat model *machine learning* yang mampu memprediksi keputusan pelanggan dalam pembelian asuransi perjalanan dengan akurasi >= 80%.

### Solution statements
Solusi dari *Problem Statements* di atas agar dapat mencapai *Goals* yang telah ditetapkan adalah menerapkan pra-pemrosesan data dan pembuatan model yang dapat diterangkan sebagai berikut:
1. Pra-pemrosesan data. Pra-pemrosesan adalah pengolahan data mentah yang bertujuan mengurangi kemungkinan model menjadi bias atau bahkan gagal mencapai akurasi yang diinginkan. Pra-pemrosesan data pada proyek ini dilakukan dengan langkah-langkah berikut:
- **Pengolahan kolom fitur** dengan memilah serta memilih kolom yang memiliki korelasi tinggi dengan kolom target.
- **Pembagian dataset** dengan data latih 80% dan data uji 20%.
- **Standarisasi data** dengan mengubah skala data menjadi relatif sama atau mendekati distribusi normal. 
2. Pembuatan model. Pembuatan model adalah menentukan model yang tepat untuk dilatih menggunakan data yang telah melewati pra-pemrosesan agar dapat memprediksi hasil sesuai tujuan proyek ini. Pembuatan model pada proyek ini menggunakan dua model yang selanjutnya model *baseline* ini akan dilakukan pengembangan untuk meningkatkan performa. Dua model tersebut adalah:
- **KNN**. KNN adalah algoritma yang menggunakan kesamaan fitur untuk memprediksi nilai baru. Nilai baru ini didasarkan pada seberapa mirip dengan tetangganya sejumlah k, oleh karena itu disebut K-Nearest Neighbor.
- **Gradient Boosting Algorithm**. Algoritma ini bekerja dengan meningkatkan (*boosting*) model yang dianggap memiliki performa rendah atau akurasi yang belum memuaskan.

## Data Understanding
(Melakukan beberapa tahapan yang diperlukan mengenai data, contohnya teknik visualisasi data atau exploratory data analysis.)
Dataset proyek ini berasal dari platform Kaggle yang dipublikasi oleh TejasTheBard dengan judul [Travel Insurance Prediction Data](https://www.kaggle.com/tejashvi14/travel-insurance-prediction-data). Berdasarkan metadata, dataset ini bersumber basis data perusahaan perjalanan di India. Dari dataset TravelInsurancePrediction.csv yang diunduh, dataset memiliki 10 kolom dengan keterangan berikut:

| Fitur               | Deskripsi                                                                                             |
| --------------------| ----------------------------------------------------------------------------------------------------- |
| Index               | Indeks atau nomor baris.                                                                              |
| Age                 | Umur pelanggan.                                                                                       |
| Employment Type     | Sektor pelanggan bekerja (Pemerintah (Government Sector) atau Swasta (Private Sector/Self Employed'). |
| GraduateOrNot       | Status lulusan perguruan tinggi.                                                                      |
| AnnualIncome        | Pendapatan tahunan (Rupee).                                                                           |
| FamilyMembers       | Jumlah anggota keluarga.                                                                              |
| ChronicDiseases     | Status ada tidaknya penyakit kronis pelanggan (asma, diabetes, darah tinggi, dll).                    |
| FrequentFlyer       | Status jika sering bepergian berdasarkan riwayat 2 tahun terakhir.                                    |
| EverTravelledAbroad | Status bepergian ke luar negeri.                                                                      |
| TravelInsurance     | Status pelanggan membeli paket asuransi.                                                              |

## Data Preparation
(Menerapkan dua atau lebih teknik yang diperlukan dalam tahapan data preparation.)
Tahap persiapan data atau pra-pemrosesan data dilakukan dengan langkah-langkah berikut:
- **Resample dataset** dengan menyeimbangkan jumlah data. Resample dataset diperlukan untuk menghindari hasil prediksi yang bias dikarenakan kuantitas yang tidak seimbang dalam sebuah data.
- **Pembagian dataset** dengan data latih 80% dan data uji 20%. Pembagian dataset tentunya diperlukan agar model yang telah dilatih dapat diujikan seberapa akurat hasil prediksinya terhadap data baru. Dalam dataset ini rasio 80:20 dapat dikatakan masih ideal karena jumlahnya masih ribuan saja (1987 baris).
- **Standarisasi data** dengan mengubah skala data menjadi relatif sama atau mendekati distribusi normal. Tahap standarisasi digunakan untuk menyeragamkan fitur numerik dalam skala data yang sama dan dapat lebih mudah diolah saat pelatihan model.

## Modeling and Result
(Menyajikan dua solusi rekomendasi dengan algoritma yang berbeda.)
Seperti yang telah dituliskan dalam *solution statement*, model machine learning yang digunakan untuk menyelesaikan permasalahan dalam proyek ini adalah KNN dan Gradient Boosting.

**1. KNN.** Model KNN proyek ini akan menggunakan library sklearn. Model dilatih dengan data yang telah melewati pra-pemrosesan. Selanjutnya akan dikembangkan model KNN ini menggunakan GridSearchCV untuk mencari hyperparameter terbaik.

**2. Gradient Boosting.** Model Gradient Boosting ini juga menggunakan library sklearn GradientBoostingClassifier dan dilatih dengan data yang telah melewati pra-pemrosesan.
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Seperti model KNN, Gradient Boosting ini juga akan dikembangkan modelnya dengan GridSearchCV untuk mencari hyperparameter terbaik.

Hasil pelatihan dan pengujian model dapat dilihat sebagai berikut:

![Hasil Model](../main/images/hasil_model.png "Hasil Model")

Dari hasil seluruh model yang dibuat, model Gradient Boosting yang dikembangkan memiliki nilai terbaik dan oleh karena itu model ini yang akan digunakan pada tahap selanjutnya.

## Evaluation
(Menjelaskan metrik evaluasi yang digunakan untuk mengukur kinerja model (formula dan cara metrik tersebut bekerja).)
Sebagai evaluasi, proyek klasifikasi akan menggunakan metrik *accuration*, *precision*, *recall*, dan *F1 score*. Kita juga akan melihat hasil *confusion matrix* dari prediksi model sebelum membahas empat metrik sebelumnya untuk lebih memberikan gambaran hasil evaluasi.
- *Confusion matrix* adalah matriks yang berisi 4 notasi tp, tn, fp, fn. Notasi tp (true positive) dan tn (true negative) menunjukkan jumlah nilai positif dan negatif yang diprediksi secara tepat. Sedangkan notasi fp (false positive) dan fn (false negative) menunjukkan jumlah nilai positif dan negatif yang diprediksi salah. Kelebihan matriks ini adalah paling sederhana untuk dipahami dan kekurangannya adalah tidak cukup informatif untuk mengukur hasil sehingga perlu diolah kembali [[3]](https://www.academia.edu/download/37219940/5215ijdkp01.pdf).
![confusion matrix](../main/images/confusion_matrix.png "confusion matrix")

- *Accuration* atau akurasi adalah salah satu metriks evaluasi kesalahan yang sering dipakai. Akurasi didapatkan dari persentase prediksi yang benar terhadap total nilai yang ada. Kelebihan metrik ini adalah penilaian yang mudah digunakan, lebih sedikit kerumitan, bisa digunakan dalam multi label atau multi kelas, dan mudah dipahami. Sedangkan kekurangannya adalah keterbatasan dalam proses evaluasi dan proses diskriminasi. Contohnya adalah jika dalam sebuah dataset dengan nilai negatif yang berjumlah 80% dan model yang dibuat memprediksi seluruhnya negatif. Maka hasil akurasinya akan tetap bernilai 80% tanpa mengetahui bahwa ada bias kesalahan prediksi, yakni model selalu memprediksi negatif.

  ![accuracy](../main/images/accuracy.png "accuracy")

- *Precision* atau presisi adalah metriks evaluasi untuk mengukur pola positif yang diprediksi dengan benar dari total pola prediksi dalam kelas positif. Kelebihan presisi adalah mampu menilai prediksi model terhadap label data positif. Ini menghasilkan presisi tidak mampu mengukur hasil label negatif.

  ![precision](../main/images/precision.png "precision")

- *Recall* adalah metriks evaluasi untuk mengukur pola positif yang diprediksi dengan benar dari total pola prediksi yang benar. Metriks ini adalah nilai yang berlawanan dengan presisi sehingga memiliki keunggulan menghitung bagian negatif dari prediksi label positif dan kekurangannya adalah tidak mampu menghitung prediksi negatif.

  ![recall](../main/images/recall.png "recall")

- *f1-score* adalah metriks evaluasi yang menggunakan nilai presisi dan *recall* untuk mengukur seberapa baik hasil dan seberapa lengkap hasil prediksinya. Kombinasi presisi dan *recall* menjadikan *f1-score* saling melengkapi kekurangan dua evaluasi metriks tersebut namun tidak dapat menghitung hasil prediksi benar pada label negatif.

  ![f1score](../main/images/f1score.png "f1score")

Seluruh evaluasi matriks ini dapat digunakan dengan mudah menggunakan *library* sklearn metrics. Hasilnya dapat dilihat pada gambar berikut:

  ![cfmatrix](../main/images/cfmatrix.png "cfmatrix")

  ![4metriks](../main/images/4metriks.png "4metriks")


## References
[[1]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.23.9764&rep=rep1&type=pdf) Swearingen, K. dan Sinha, R., 2001, September. Beyond algorithms: An HCI perspective on recommender systems. In*ACM SIGIR 2001 workshop on recommender systems* (Vol. 13, No. 5-6, pp. 1-11). [http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.23.9764&rep=rep1&type=pdf](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.23.9764&rep=rep1&type=pdf)

[[2]](https://ejournal.akprind.ac.id/index.php/technoscientia/article/view/612) Irfan, M. dan Cahyani, A.D., 2014. Sistem Rekomendasi: Buku Online Dengan Metode Collaborative Filtering. *Jurnal Teknologi Technoscientia*, pp.076-84. [https://ejournal.akprind.ac.id/index.php/technoscientia/article/view/612](https://ejournal.akprind.ac.id/index.php/technoscientia/article/view/612)

[[3]](https://www.oecd.org/education/pisa-2018-results-volume-i-5f07c754-en.htm) OECD. (2019). *PISA 2018 results (volume I): what students know and can do.* PISA:
OECD Publishing. [https://www.oecd.org/education/pisa-2018-results-volume-i-5f07c754-en.htm](https://www.oecd.org/education/pisa-2018-results-volume-i-5f07c754-en.htm)

**---Ini adalah bagian akhir laporan---**
