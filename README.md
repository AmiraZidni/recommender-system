# Proyek Machine Learning - Recommender System 

## Table of contents

- [Project Overview](#project-overview)
- [Business Understanding](#business-understanding)
- [Data Understanding](#data-understanding)
- [Data Preparation](#data-preparation)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [References](#references)


## Project Overview
Proyek *machine learning* ini akan membahas tentang **"Sistem Rekomendasi Buku dengan Content Based Filtering dan Collaborative Filtering"**.

![banner](https://user-images.githubusercontent.com/68690376/139519652-2bc91225-c5f7-4db2-a238-fd25d1d707d5.png)

Membaca buku adalah salah satu kegiatan yang belum banyak peminatnya di Indonesia. Padahal membaca adalah jendela ilmu. Bahkan salah satu segmen pasar awal e-commerce Amazon adalah pasar buku sebelum akhirnya meluaskan target pasar ke segmen lainnya. Dalam e-commerce Amazon, tentunya terdapat sistem rekomendasi yang membantu pengunjung mencari buku agar pengunjung website tersebut membuat keputusan akhir yaitu membelinya [[1]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.23.9764&rep=rep1&type=pdf). Sistem rekomendasi buku juga dapat membantu pembaca buku mengetahui informasi buku-buku yang akan dibaca selanjutnya sesuai dengan preferensi pengguna dari rekam jejak buku yang telah dibaca sebelumnya. Terjadi keuntungan dua pihak karena konsumen mendapatkan informasi yang diperlukan untuk membuat suatu keputusan sementara dari pemilik bisnis juga mendapatkan profit dari penjualan. Hal ini menjadikan sistem rekomendasi telah digunakan secara luas oleh hampir semua area bisnis. [[2]](https://ejournal.akprind.ac.id/index.php/technoscientia/article/view/612).

Pada proyek machine learning ini, sistem rekomendasi yang akan dibuat memiliki fokus pada sistem rekomendasi buku karena di Indonesia sendiri masih diperlukan sebuah sistem yang dapat membantu merekomendasikan para pembaca agar lebih mudah mendapatkan informasi buku-buku yang akan dibaca selanjutnya. Hal ini didasarkan pada hasil survey kemampuan membaca oleh PISA tahun 2018 yang menghasilkan nilai skor membaca pelajar Indonesia yang masih sangat rendah [[3]](https://www.oecd.org/education/pisa-2018-results-volume-i-5f07c754-en.htm).

## Business Understanding
Membaca adalah salah satu kegiatan mengumpulkan informasi serta memahami makna dari sebuah tulisan. Manfaat membaca diantaranya adalah meningkatkan kapasitas berpikir dan meningkatkan keterampilan menulis. Selain itu, membaca juga dapat menjadi kegiatan hiburan atau kegiatan yang mampu menenangkan pikiran karena sebuah bacaan dapat membawa pembacanya ke dunia imajinasi luas dalam konteks atau tema yang diangkat. Sehingga tak jarang apabila setelah membaca buku timbul keinginan untuk melanjutkan ke bacaan lainnya yang memiliki relasi atau kemiripan dengan bacaan sebelumnya.

![journey](https://img.freepik.com/free-psd/every-book-is-new-wonderful-travel-quote-with-books_23-2148355614.jpg?size=626&ext=jpg)

Keinginan ini membuat pembaca buku akan mencari preferensi buku bacaan selanjutnya yang serupa atau memiliki relasi terhadap buku yang telah dibaca. Relasi atau kesamaan tersebut dapat berupa kelanjutan isi buku atau kesamaan penulis buku. Harapannya adalah agar pembaca mendapatkan informasi lanjutan atau kesamaan sensasi membaca dari buku yang akan dibaca selanjutnya. Sistem rekomendasi dapat menjadikan salah satu solusi dari keperluan pembaca untuk mencari buku selanjutnya karena sistem rekomendasi mampu menghasilkan informasi-infomasi buku yang memiliki kemiripan dengan suatu preferensi tertentu. 

### Problem Statements

Dari latar belakang di atas, dapat ditarik rumusan masalah sebagai berikut:
1. Sistem rekomendasi apa yang baik untuk diterapkan dalam konteks pemberian rekomendasi buku?
2. Bagaimana cara membuat sistem rekomendasi bagi pembaca buku?

### Goals
Tujuan proyek yang ingin dicapai adalah:
1. Menentukan sistem rekomendasi buku yang hasilnya relevan dengan pembaca buku.
2. Membuat sistem rekomendasi buku bagi pembaca buku.

### Solution Approach
Pendekatan solusi untuk mencapai *Goals* di atas dapat dijabarkan dalam langkah-langkah berikut:
1. Mengunduh data tentang rekomendasi buku.

    Data diunduh dari Kaggle dengan tautan berikut [Book Recommendation Dataset](https://www.kaggle.com/arashnic/book-recommendation-dataset).
2. Melakukan pra-pemrosesan data.

    Pra-pemrosesan data diperlukan agar data yang akan diumpankan pada model *machine learning* tidak memiliki bias atau mengalami kegagalan memprediksi rekomendasi. Pra-pemrosesan data pada proyek ini adalah:
      - **Mengganti nama kolom** dengan nama yang bisa digunakan dalam pengolahan lanjutan yakni mengubah strip (-) dengan garis bawah (_).
      - **Memperbaiki nilai** dari baris yang terdapat kesalahan ketik atau nilai abnormal.
      - **Menghapus kolom yang tidak diperlukan** seperti kolom gambar karena tidak berpengaruh ke pembuatan sistem rekomendasi.
      - **Menghapus baris** yang memiliki nilai kosong.
3. Melakukan persiapan data.

    Persiapan data pada proyek ini dilakukan untuk mempersiapkan data sebelum digunakan untuk melatih model sistem rekomendasi. Persiapan data diantaranya adalah:
      - **Menggabungkan data** berdasarkan nilai kolom kunci dan **menghapus baris** yang tidak memiliki nilai setelah digabungkan.
      - **Mengelompokkan dan mengurutkan data** berdasarkan *rating*.
      - **Mengambil sampel data** karena data ini terlalu besar dan menghabiskan banyak sumber daya RAM yang dapat menghasilkan *notebook crash* pada Google Colab proyek ini.
      - **Pembagian dataset** dengan data latih 80% dan data uji 20%.
4. Membangun sistem rekomendasi.

    Sistem rekomendasi yang diajukan untuk menyelesaikan permasalahan ini adalah penggunaan dua algoritma *Content Based Filtering* dan *Collaborative Filtering*.
    - **Content Based Filtering**. Dalam konteks penelitian ini dan berdasarkan sumber [[2]](https://ejournal.akprind.ac.id/index.php/technoscientia/article/view/612), algoritma *content based filtering* bekerja dengan mencari kedekatan suatu buku yang akan direkomendasikan dengan buku yang telah diambil oleh pembaca berdasarkan kemiripan antar isinya.
        - Kelebihan:
            - Hasil rekomendasi didasarkan pada preferensi buku.
            - Sederhana dan transparan karena mudah dipahami bagaimana algoritma ini bekerja.
        - Kelemahan:
            - Pembaca buku tidak mendapatkan rekomendasi dari jenis buku yang berbeda.
    - **Collaborative Filtering**. Algoritma *collaborative filtering* dalam konteks penelitian ini bekerja dengan mengumpulkan dan mengolah sejumlah besar informasi yang didasarkan pada aktifitas pembaca buku seperti pemberian nilai, penulis buku, atau preferensi lainnya. Informasi yang telah dikumpulkan dan diolah tersebut akan digunakan sebagai preferensi dengan pembaca buku lain dengan mencari kesamaannya.
        - Kelebihan:
            - Informasi tentang preferensi dapat ditambahkan secara mudah.
            - Adanya rekam jejak preferensi memudahkan sistem rekomendasi bekerja lebih baik.
        - Kelemahan:
            - Kurang efektif terhadap pengguna yang belum memiliki data karena tidak terdapat informasi yang cukup.
5. Melakukan evaluasi hasil sistem rekomendasi.

    Tahap ini dilakukan untuk mengukur seberapa baik hasil sistem rekomendasi. Berdasarkan algoritma yang akan digunakan, evaluasi yang bisa diterapkan adalah nilai presisi >80% pada algoritma *content based filtering* dan nilai RMSE (*Root Mean Squared Error*) <10% skala data pada algoritma *collaborative filtering*.

## Data Understanding
Dataset proyek ini berasal dari platform Kaggle yang dipublikasi oleh Möbius dengan judul [Book Recommendation Dataset](https://www.kaggle.com/arashnic/book-recommendation-dataset). Berdasarkan metadata, dataset ini dikumpulkan oleh Cai-Nicolas Ziegler di tahun 2004 yang bersumber dari komunitas [bookcrossing](
https://www.bookcrossing.com/howto). Terdapat tiga buah dataset yaitu Books.csv, Ratings.csv, dan Users.csv.

Books.csv adalah data mengenai buku dengan 8 kolom dengan keterangan sebagai berikut:

| Kolom               | Deskripsi                                                                                      |
| --------------------| ---------------------------------------------------------------------------------------------- |
| ISBN                | *International Standard Book Number* atau deretan 13 digit angka nomor identifikasi buku.      |
| Book-Title          | Judul buku.                                                                                    |
| Book-Author         | Penulis Buku.                                                                                  |
| Year-Of-Publication | Tahun publikasi buku.                                                                          |
| Publisher           | Penerbit buku.                                                                                 |
| Image-URL-S         | Alamat URL cover buku ukuran kecil.                                                            |
| Image-URL-M         | Alamat URL cover buku ukuran menengah.                                                         |
| Image-URL-L         | Alamat URL cover buku ukuran besar.                                                            |

Umumnya kita bisa langsung melihat visualisasi distribusi penyebaran nilai dari kolom tahun terbit (Year-Of-Publication) tetapi setelah diperiksa terdapat kesalahan tipe data pada kolom tersebut yang seharusnya int tetapi justru *object*. Ada kemungkinan terdapat kesalahan nilai di dalamnya yang akan kita periksa di proses selanjutnya.

![wrongtypebookyear](https://user-images.githubusercontent.com/68690376/139523432-0f11289c-5785-4a50-8f07-e018bbcc9c1b.png)

Dalam dataset Ratings.csv yang berisi informasi peringkat buku terdapat 3 kolom dengan keterangan berikut:

| Kolom               | Deskripsi                                                                                        |
| --------------------| ------------------------------------------------------------------------------------------------ |
| User-ID             | Nomor identifikasi *user* atau pemberi nilai buku.                                                    |
| ISBN                | *International Standard Book Number* atau deretan 13 digit angka nomor identifikasi buku.        |
| Book-Rating         | Nilai buku yang diberikan *user* atau pemberi nilai buku. Berisi rentang nilai 0-10 (semakin tinggi semakin baik) sebagai penilaian eksplisit dan juga terdapat nilai 0 sebagai penilaian implisit.              |

Distribusi Book-Rating memiliki nilai 0 cukup banyak. Menurut metadata, nilai 0 ini adalah penilaian implisit yang perlu dilakukan pengolahan di proses selanjutnya untuk membedakan penilaian eksplisit.

![bookrating](https://user-images.githubusercontent.com/68690376/139523475-e8aea4bd-c352-40ea-ace7-0c82e35786bf.png)


Terakhir, dalam dataset Users.csv yang berisi informasi tentang *user* atau pemberi nilai buku terdapat 3 kolom dengan keterangan berikut:

| Kolom               | Deskripsi                                                                            |
| --------------------| ------------------------------------------------------------------------------------ |
| User-ID             | Nomor identifikasi *user* atau pemberi nilai buku.                                        |
| Location            | Lokasi *user* atau pemberi nilai buku yang tersusun dari kota, negara bagian, dan negara. |
| Age                 | Usia *user* atau pemberi nilai buku.                                                      |

Distribusi nilai Age pada data Users.csv memiliki skewness positif dan terdapat nilai abnormal yang akan kita proses di tahap selanjutnya.
![rawagegraph](https://user-images.githubusercontent.com/68690376/139523117-59349e35-a01a-493c-b6e4-845f4af639a3.png)

## Data Preparation
Tahap persiapan data sebagaimana yang telah dijelaskan pada bagian *Solution Approach* terdiri dari langkah-langkah berikut:
- **Mengganti nama kolom.**

    Pergantian nama ini dilakukan karena beberapa nama kolom seperti User-ID akan sulit digunakan dalam penggunaan syntax python di langkah selanjutnya. Pergantian nama kolom dilakukan dengan cara mengubah karakter strip (-) dengan garis bawah (_). Nama-nama kolom yang diubah adalah:
        1. Book-Title menjadi book_title.
        2. Book-Author menjadi book_author.
        3. Year-Of-Publication menjadi year_of_publication.
        4. User-ID menjadi user_id.
        5. Book-Rating menjadi book-rating.
- **Memperbaiki nilai.**

    Hal ini dilakukan karena beberapa isi kolom data terdapat kesalahan ketik atau nilai abnormal seperti:
        - Nilai kolom Year-Of-Publication pada data Books.csv yang memiliki kesalahan nilai dan terdapat nilai abnormal 0 dan nilai di atas 2014 (dataset diambil tahun 2014). Perbaikan nilai pada kesalahan tulis dapat diperbaiki secara manual. Sedangkan nilai abnormal diperbaiki dengan melakukan imputasi nilai yang sering muncul atau nilai modus karena imputasi dengan nilai mean tidak cocok untuk kolom ini yang mana memiliki skewness negatif yang apabila diberikan imputasi mean akan mengubah pola distribusinya.
        
    
    ![wrongyearvalue](https://user-images.githubusercontent.com/68690376/139519636-3887c3f7-4824-485f-a0d1-01020f8820fd.png)
    ![wrongbookvalue](https://user-images.githubusercontent.com/68690376/139523219-a4ea34a5-96d4-4e4e-9c07-2542597c9741.png)

    Gambar diatas adalah penampakan kesalahan nilai pada kolom Year-Of-Publication. Setelah dilakukan perbaikan, dapat dilakukan visualisasi kolom Year-Of-Publication untuk menampilkan distribusi nilainya.

    ![afterprocessyearskewness](https://user-images.githubusercontent.com/68690376/139523249-c7a4bb9d-44a9-4afb-b3b5-b018e30a4156.png)



    Selanjutnya nilai kolom Age pada Users.csv yang memiliki nilai abnormal 0 dan di atas 95 tahun. Hal ini menjadi abnormal karena secara nalar tidak mungkin seseorang yang berumur kurang dari 5 tahun atau lebih dari 95 tahun memberikan penilaian buku. Nilai abnormal kolom pada ini diberlakukan sama dengan teknik sebelumnya yakni imputasi dengan nilai modus karena kolom ini memiliki skewness positif.
        
    ![wrongagevalue](https://user-images.githubusercontent.com/68690376/139520522-7353c295-d5de-415d-9ba7-5e52f016ce68.png)

- **Menghapus kolom yang tidak diperlukan.**

    Kolom atau fitur pada data yang tidak digunakan seperti Image-URL-S, Image-URL-M, Image-URL-L akan dihapus karena pembangunan sistem rekomendasi tidak membutuhkan fitur alamat URL buku.
- **Menghapus baris yang memiliki nilai kosong.**

    Penghapusan baris dengan ketentuan tertentu seperti baris yang tidak memiliki nilai dilakukan setelah perbaikan nilai pada tahap sebelumnya tidak memungkinan untuk dilakukan. Beberapa penghapusan baris dengan ketentuannya dilakukan pada baris berikut:
        - Baris yang memiliki nilai kosong Book-Author dan Publisher dalam data Books.csv.
        - Baris yang memiliki nilai 0 pada kolom book_rating dalam Users.csv karena saat ini belum cukup informasi untuk mengolah penilaian implisit.
- **Menggabungkan data.**

    Penggabungan data dilakukan untuk mengumpulkan seluruh data berdasarkan fitur user_id dan ISBN. Hasil penggabungan ini adalah sebuah data utama yang perlu dibersihkan kembali pada baris yang memiliki nilai kosong sebelum melanjutkan langkah berikutnya.
- **Mengelompokkan dan mengurutkan data.**

    Hal ini dilakukan dengan menghimpun data utama yang telah digabung dan diurutkan berdasarkan book_rating agar mengembalikan data utama yang bersifat unik dan urut berdasarkan penilaian buku tertinggi.
- **Mengambil sampel data.**

    Langkah ini diperlukan karena data yang sudah dibersihkan dan diolah memiliki ukuran terlalu besar dan akan menghabiskan banyak sumber daya RAM yang dapat menghasilkan *notebook crash* atau berhentinya proses komputasi pada Google Colab dalam proyek ini. Ukuran sampel yang diambil adalah 10.000 baris pertama dari data utama.
 
- **Pembagian dataset.**
    Pembagian ditentukan dengan ukuran data latih 80% dan data uji 20%. Ini diperlukan agar model yang telah dilatih dapat diujikan seberapa akurat hasil prediksinya terhadap data baru. Dalam dataset ini rasio 80:20 dapat dikatakan masih ideal karena jumlahnya masih ribuan saja (1987 baris). !!!

## Modeling
Tahapan ini membahas mengenai **pembuatan model sistem rekomendasi** untuk menyelesaikan permasalahan dan **menyajikan top-N recommendation sebagai solusi.**

Untuk menjelaskan mengenai bagian ini, Anda dapat mengikuti panduan: 
- Jelaskan bagaimana Anda melakukan proses modeling dalam proyek. 
- Sajikan top-N recommendation sebagai output model Anda.
- Jelaskan pula hasil rekomendasi dari model Anda.

(Menyajikan dua solusi rekomendasi dengan algoritma yang berbeda.)
Seperti yang telah dituliskan dalam *solution statement*, model machine learning yang digunakan untuk menyelesaikan permasalahan dalam proyek ini adalah KNN dan Gradient Boosting.

**1. KNN.** Model KNN proyek ini akan menggunakan library sklearn. Model dilatih dengan data yang telah melewati pra-pemrosesan. Selanjutnya akan dikembangkan model KNN ini menggunakan GridSearchCV untuk mencari hyperparameter terbaik.

**2. Gradient Boosting.** Model Gradient Boosting ini juga menggunakan library sklearn GradientBoostingClassifier dan dilatih dengan data yang telah melewati pra-pemrosesan.
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Seperti model KNN, Gradient Boosting ini juga akan dikembangkan modelnya dengan GridSearchCV untuk mencari hyperparameter terbaik.

Hasil pelatihan dan pengujian model dapat dilihat sebagai berikut:

![Hasil Model](../main/images/hasil_model.png "Hasil Model")

Dari hasil seluruh model yang dibuat, model Gradient Boosting yang dikembangkan memiliki nilai terbaik dan oleh karena itu model ini yang akan digunakan pada tahap selanjutnya.

## Evaluation
Bagian ini menjelaskan mengenai metrik evaluasi yang digunakan untuk mengukur kinerja model.  Penjelasannya meliputi (namun tidak terbatas pada) beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan dan bagaimana formulanya
- Kelebihan dan kekurangan metrik
- Bagaimana cara menerapkannya ke dalam kode.

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
