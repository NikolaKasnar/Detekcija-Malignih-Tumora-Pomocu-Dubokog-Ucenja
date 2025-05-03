# Projekt: Detekcija Malignih Tumora Pomoću Dubokog Učenja na Medicinskim Slikama

## 1. Uvod

Strojni vid, iako povijesno izazovno područje, doživio je značajan napredak u posljednjem desetljeću, posebice zahvaljujući razvoju dubokog učenja. Njegova primjena u zdravstvu otvara nove mogućnosti za dijagnostiku i analizu medicinskih podataka. Ovaj projekt fokusirao se na primjenu suvremenih metoda strojnog vida za rješavanje problema automatske klasifikacije tumora kao benignih ili malignih na temelju njihovih medicinskih slika.

## 2. Podaci i Predobrada

Osnovu projekta činile su medicinske slike pohranjene u **DICOM (.dcm) formatu**. Primarni fokus analize bio je na **slikovnim podacima (pixel_data)** sadržanim unutar ovih datoteka, iako je inicijalno razmatrana i mogućnost korištenja pridruženih **metapodataka**.

Rad s medicinskim slikama donio je specifične izazove:
- **Neuravnoteženost klasa:** Inicijalni skup podataka pokazao je izrazitu neuravnoteženost, gdje su maligni tumori činili svega 2% ukupnih primjera. Ovo predstavlja značajan rizik da model nauči trivijalno rješenje (uvijek predviđa benigni tumor).
- **Velika količina podataka:** Velik broj slika otežavao je učitavanje i obradu unutar dostupne radne memorije (RAM).

Kako bismo riješili problem sporog učitavanja i ograničenja memorije, podatke (pixel_data) smo pretvorili i pohranili u **LMDB (Lightning Memory-Mapped Database)** format. Ovaj format je optimiziran za vrlo brzo čitanje podataka, što je ključno za efikasno treniranje modela.
Ključni korak u predobradi bio je rješavanje problema neuravnoteženosti klasa. Usvojili smo strategiju **oversamplinga manjinske klase**: tijekom formiranja svake serije (batcha) podataka za treniranje, primjeri malignih tumora birani su višestruko.

## 3. Metodologija

### 3.1. Modeliranje

3.1. Modeliranje
Za klasifikaciju tumora korištene su **duboke konvolucijske neuronske mreže (CNN)**. 

Primijenjen je pristup **transfernog učenja (transfer learning)**:
1. Korišteni su modeli s arhitekturama koje su se pokazale uspješnima na općim zadacima računalnog vida (npr. ResNet, EfficientNet, itd.), prethodno trenirani na velikim skupovima podataka poput ImageNeta. Ideja je da su ovi modeli već naučili prepoznavati osnovne vizualne značajke (rubove, teksture, oblike).
2. Na predtreniranu osnovu modela ("backbone") dodan je prilagođeni klasifikacijski dio ("head"), koji se obično sastoji od nekoliko potpuno povezanih slojeva (fully connected layers).
3. Ovakva podjela na "backbone" i "head" omogućila je primjenu različitih strategija treniranja za svaki dio.

### 3.2. Treniranje

Proces treniranja uključivao je pažljiv odabir hiperparametara i tehnika:

**Augmentacija podataka:** Kako bi se smanjila prenaučenost (overfitting), posebno zbog višestrukog korištenja malignih uzoraka, primijenjene su intenzivne augmentacije podataka pomoću biblioteke Albumentations. Korištene transformacije uključivale su (s primjerima vjerojatnosti primjene p):
- Geometrijske: Transpozicija (p=0.5), vertikalno (p=0.5) i horizontalno zrcaljenje (p=0.5), nasumična rotacija, skaliranje i pomak (ShiftScaleRotate, p=0.85), optička distorzija (p=0.7), distorzija mreže (GridDistortion, p=0.7), elastična transformacija (ElasticTransform, p=0.7).
- Boja i Kontrast: Nasumične promjene svjetline i kontrasta (RandomBrightnessContrast, p=0.75), promjene nijanse, zasićenosti i osvjetljenja (HueSaturationValue, p=0.5), adaptivno izjednačavanje histograma (CLAHE, p=0.7).
- Zamućenje i Šum: Različite vrste zamućenja (Motion, Median, Gaussian) ili Gaussov šum (p=0.7).
- Regularizacija: Izrezivanje većih ili manjih pravokutnih dijelova slike (CoarseDropout, p=1.0).
- Sve slike su prije ulaska u model skalirane na definiranu veličinu (image_size) i normalizirane.

**Funkcije gubitka (Loss Functions):** Za mjerenje pogreške modela korištene su BCEWithLogitsLoss (standardna i numerički stabilna funkcija za binarnu klasifikaciju) i Focal Loss (posebno korisna kod neuravnoteženih skupova jer daje veću težinu primjerima koje model teško klasificira, često onima iz manjinske klase).

**Optimizatori i Stopa Učenja (Learning Rate):** Eksperimentiralo se s različitim optimizatorima, uključujući **AdamW** i **SGD s momentumom**. Isprobane su različite strategije za prilagodbu stope učenja tijekom treniranja: konstanta stopa, **OneCycleLR**, **Cosine Annealing** i **ReduceLROnPlateau** (smanjenje stope učenja kada se metrika na validacijskom skupu prestane poboljšavati). Primijenjene su **diferencijalne stope učenja**: "backbone" modela treniran je sa sporijom stopom učenja nego "head". Također, u početnim fazama treniranja, težine "backbone"-a su bile "**zamrznute**" (nisu se ažurirale) kako bi se prvo prilagodio samo novododani "head".

**Regularizacija:** U klasifikacijski "head" modela dodavane su tehnike regularizacije poput **Batch Normalization** i **Dropout** kako bi se dodatno smanjila prenaučenost.
- Dinamičke stope učenja (OneCycleLR, Cosine Annealing)

## 4. Evaluacija

Kvaliteta modela primarno je mjerena pomoću metrike **AUC ROC (Area Under the Receiver Operating Characteristic Curve)** na izdvojenom **testnom skupu podataka**, koji model nije vidio tijekom treniranja ili validacije. Praćenjem vrijednosti funkcije gubitka na skupu za treniranje i AUC metrike na validacijskom skupu (koji je korišten za dinamičko podešavanje stope učenja i rano zaustavljanje) nastojalo se balansirati između performansi i prenaučenosti.

## 5. Rezultati i Ključna Saznanja

- Najveći pomak u performansama postignut je implementacijom **višestrukog uzorkovanja (oversamplinga) malignih tumora** u kombinaciji s jakim augmentacijama. Bez toga, model je težio predviđati samo dominantnu (benignu) klasu, što je rezultiralo niskom vrijednošću funkcije gubitka, ali i niskim AUC ROC rezultatom.
- Modeli trenirani s **dinamičkim rasporedom stope učenja**, poput ReduceLROnPlateau temeljenom na validacijskom AUC, pokazali su se uspješnima, no zahtijevali su pažljivo praćenje kako bi se izbjeglo prenaučavanje na validacijski skup. Korištenje diferencijalnih stopa učenja i početno zamrzavanje "backbone"-a također su doprinijeli stabilnosti treniranja.
- Kao konačni korak, najbolji pojedinačni modeli kombinirani su u **ansambl model**. Isprobane su metode poput usrednjavanja predikcija, odabira najsigurnije predikcije i treniranja meta-modela (stacking) na izlazima pojedinačnih modela. (Potrebno je dodati koja se metoda pokazala najboljom).
- Najbolji postignuti rezultat na testnom skupu, koristeći ansambl pristup, bio je **AUC ROC = 0.935**.

## 6. Izazovi
Glavni praktični izazov bio je **dugo trajanje treniranja** pojedinačnih modela, koje je često trajalo i više od jednog dana. Ovo je značajno usporilo proces iteranja kroz različite hiperparametre i arhitekture, zahtijevalo strpljenje i činilo svaku grešku u postavkama vremenski skupom.

## 7. Zaključak

Projekt je uspješno demonstrirao primjenu dubokog učenja i transfernog učenja na zahtjevnom medicinskom zadatku klasifikacije tumora iz DICOM slika. Kroz sustavnu primjenu tehnika poput oversamplinga neuravnoteženih klasa, intenzivne augmentacije podataka, pažljivog odabira funkcija gubitka, optimizatora (AdamW, SGD) i strategija učenja (ReduceLROnPlateau, diferencijalne stope učenja, zamrzavanje), te korištenjem ansambl metoda, postignut je visok rezultat od **0.935 AUC ROC** na testnom skupu. 
