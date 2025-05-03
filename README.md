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

**Augmentacija podataka**: Kako bi se smanjila prenaučenost (overfitting), posebno zbog višestrukog korištenja malignih uzoraka, primijenjene su intenzivne augmentacije podataka pomoću biblioteke Albumentations. Korištene transformacije uključivale su (s primjerima vjerojatnosti primjene p):
- ○	Geometrijske: Transpozicija (p=0.5), vertikalno (p=0.5) i horizontalno zrcaljenje (p=0.5), nasumična rotacija, skaliranje i pomak (ShiftScaleRotate, p=0.85), optička distorzija (p=0.7), distorzija mreže (GridDistortion, p=0.7), elastična transformacija (ElasticTransform, p=0.7).
- Promjene boja i kontrasta
- Dodavanje šuma
- Regularizacijske tehnike

**Funkcije gubitka:**
- BCEWithLogitsLoss
- Focal Loss

**Optimizatori:**
- AdamW
- SGD s momentumom

**Strategije učenja:**
- Dinamičke stope učenja (OneCycleLR, Cosine Annealing)
- Diferencijalne stope učenja
- Zamrzavanje težina "backbone"-a u početnim fazama

## 4. Evaluacija

Glavna metrika: **AUC ROC**

Postignuti rezultati:
- Najbolji pojedinačni modeli: AUC ROC ~0.92
- **Ansambl modela: AUC ROC = 0.935**

## 5. Zaključak

Projekt je uspješno demonstrirao primjenu dubokog učenja za klasifikaciju tumora, postižući vrhunske rezultate (AUC ROC 0.935) kroz:
- Pažljivu obradu neuravnoteženih podataka
- Napredne tehnike augmentacije
- Optimalan odabir modela i strategija učenja
- Korištenje ansambl metoda
