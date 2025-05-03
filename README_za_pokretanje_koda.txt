Kod za pokretanje se nalazi u rjesenje.py. Za uspješno pokretanje potrebno je upisati kako je trazeno na prezentaciji "rjesenje.py <INPUT_FOLDER> <OUTPUT_CSV›" ili 
se i path do trazenih foldera moze unesti rucno u kodu kod dijela:

folder_path = sys.argv[1]
output_path = sys.argv[2]

(mora se ipak unesti rucno)

U folderu modeli se nalaze 2 modela. model_2 radi u modelu CoConvNeXtBinaryClassifier bez metadate, a model_1 u modelu ConvNeXtBinaryClassifier_metadata sa 
metadatom te je to potrebno promijeniti u kodu koji zelite koristiti.

U zip datoteki originalni_kod se nalazi kod sa kojim smo dobili rjesenja preko duzih treniranja preko noci te ih tu ostavljamo za svaki slucaj. Jedna napomena je da
se u CNN_main_notebook_modified.ipynb nalazi kod koji je potrebno pokrenuti za prvotnu pretvorbu podataka iz .cmd oblika u .lmdb za koji smo primijetili da je puno pogodniji
za ucitavanje podataka pri treniranju te puno brzi.

anatom_site_collumns_train.npy nam sluzi za mapiranje dijelova tijela u one hit encoding.