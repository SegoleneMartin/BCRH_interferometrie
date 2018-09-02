# BCRH_interferometrie

Résumé du contenu des codes:

> CODES

	•	Bibliotheque_fonctions.py : ma propre bibliothèques de fonction contenant les fonctions GeneVectGaussien, Cov_y, extraction_phase, fonction_minimisee. On y fait régulièrement appel dans les autres codes.




	> BCRH_Guarnieri_Tebaldini

	•	BCRH_Guarnieri_Tebaldini.py : Ce code a pour but de reproduire les figures de l'article "Hybrid Cramer-Rao Bounds for Crustal Displacement Field Estimators in SAR Interferométry" : (SECTION 1) borne en fonction de N pour différents L, (SECTION 2) borne pour différentes corrélations, (SECTION 3) comparaison avec la borne déterministe sur v, (SECTION 4) extension à 2 paramètres déterministes pour des baseline tirés uniformément ou avec une loi normale.
	
	•	extraction_phase.py : Ce code propose l'évaluation de la qualité de l'extraction des différences de phases à partir des images. (SECTION 1) Presentation de plusieurs méthodes d'extraction, (SECTION 2) A N fixé, BCR sur chacun des \Psi_n ainsi que la variance de leur estimation, (SECTION 3) Plot pour différents L de la borne + variance estimation par methode itérative pour chaque \Psi_n, (SECTION 4) Evolution variance et biais sur chacun des \Psi_n en fonction de L pour la méthode itérative 
	
	•	BCRH+MV_Tebaldini.py : Ce code reconstitue toute les étapes du papier de Tebaldini: 
1) calcul de la borne, 
2) simulation des images, 
3) extraction des phases, 
4) calcul du MV sur theta
(SECTION 1): Calcul borne et Variance estimateur MV et MVMAP en fonction de N. A chaque fois on enregistre la figure sous le nom 'MV_N_L=*' et on stocke les liste dans le fichier npy du même nom.
(SECTION 2): Calcul borne et Variance estimateur MV et MVMAP en fonction de L. A chaque fois on enregistre la figure sous le nom 'MV_L_N=*' et on stocke les liste dans le fichier npy du même nom.
(SECTION 3): MV (ou MVMAP au choix) en générant directement les phases, en fonction de N.
(SECTION 4): MV (ou MVMAP au choix) en générant directement les phases, en fonction de L.

	•	BCRH+MVMAP_Tebaldini : Ce code fait en gros la même chose que le code BCRH+MV_Tebaldini sauf qu’il plote la borne sur les omega également.




	> BCRH_Ameliorations

	⁃	bornes.py: Ce code a pour but de ploter les différentes bornes (X de taille (N x N) ou de taille ((N-1) x (N-1)), psuedo inverse, borne deterministe...) pour les comparer.
	
	⁃	generation_APS.py : Ce code modélise l'APS sur la fenêtre Omega (taille l1 x l2) de N images et fait un plot
	
	⁃	BCRH_Ameliorations.py :Ce code propose plusieurs améliorations de la BCRH donnée par l'article "Hybrid Cramer-Rao Bounds for Crustal Displacement Field Estimators in SAR Interferométry"
SECTION 1: ajout d'une correlation spatiale dans les images
SECTION 2: ajout d'une variation lente de l'APS en plus de la corrélation spatiale

		
