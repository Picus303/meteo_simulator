# Modèle atmosphérique barotrope (1 couche) — Physique & Maths

> **But** — Définir un modèle 2D "colonne intégrée" (shallow‑water barotrope) avec cycle de l’eau minimal et bilan énergétique propre, utilisable sur n’importe quelle planète (paramètres agnostiques).

---

## Table des matières

1. [Hypothèses & portée](#hypothèses--portée)
2. [Variables, constantes & unités](#variables-constantes--unités)
3. [Équations fondamentales](#équations-fondamentales)

   * 3.1 [Masse de colonne](#31-masse-de-colonne)
   * 3.2 [Quantité de mouvement (SW)](#32-quantité-de-mouvement-sw)
   * 3.3 [Eau atmosphérique (vapeur, nuage)](#33-eau-atmosphérique-vapeur-nuage)
   * 3.4 [Thermodynamique (température potentielle)](#34-thermodynamique-température-potentielle)
4. [Rayonnement & astronomie](#rayonnement--astronomie)

   * 4.1 [Géométrie solaire & orbite](#41-géométrie-solaire--orbite)
   * 4.2 [Courte longueur d’onde (SW)](#42-courte-longueur-donde-sw)
   * 4.3 [Grande longueur d’onde (LW)](#43-grande-longueur-donde-lw)
5. [Saturation & thermodynamique humide](#saturation--thermodynamique-humide)
6. [Pression à altitude fixe (hypsométrique)](#pression-à-altitude-fixe-hypsométrique)
7. [Diagnostics & bilans](#diagnostics--bilans)
8. [Orographie simple & conservative](#orographie-simple--conservative)
9. [Paramètres planète‑agnostiques — pense‑bête](#paramètres-planèteagnostiques--pensebête)
10. [Extensions possibles](#extensions-possibles)

---

## Hypothèses & portée

* **Barotrope 1 couche** (pas de structure verticale explicite).
* **Gaz sec dominant** + **un condensable** (ex. vapeur d’eau).
* **Rayonnement gris** (paramétrique) et **cycle de l’eau minimal** (vapeur ↔ nuage ↔ pluie).
* **Surface passive**: on tient la comptabilité énergétique via un compteur d’énergie de surface $E_{\rm sfc}$ sans pronostiquer $T_s$.
* **Géométrie sphérique**; toutes les intégrales sont ponderées par l’aire.

---

## Variables, constantes & unités

* Coordonnées: latitude $\phi$, longitude $\lambda$, temps $t$.
* Vitesse horizontale: $\mathbf u=(u,v)$ \[m·s⁻¹].
* Masse de colonne: $M$ \[kg·m⁻²]; pression de surface: $p_s=gM$ \[Pa].
* Capacité thermique (colonne): $C$ \[J·m⁻²·K⁻¹].
* Traceurs: vapeur $q$ \[kg/kg], eau nuageuse $c$ \[kg/kg].
* Gravité $g$ \[m·s⁻²], rotation $\Omega$ \[s⁻¹], Coriolis $f=2\Omega\sin\phi$ \[s⁻¹].
* Constantes thermodynamiques: $R_d$ (gaz sec), $R_v$ (condensable), $c_p$, $L$ (chaleur latente), $\sigma$ (Stefan–Boltzmann).
* Diffusions effectives: $\nu$ (moment), $\kappa_T$ (température), $\kappa_q,\kappa_c$ (traceurs).

---

## Équations fondamentales

### 3.1 Masse de colonne

$$
\partial_t M + \nabla\!\cdot(M\,\mathbf u) = S_M,
\qquad
S_M=\frac{M_{\rm diag}-M}{\tau_M}-\Big\langle\frac{M_{\rm diag}-M}{\tau_M}\Big\rangle_A.
$$

* Conservation globale stricte si $\langle S_M\rangle_A=0$.
* Pression de surface: $\boxed{\ p_s=gM\ }$.

### 3.2 Quantité de mouvement (SW)

Forme conservative barotrope (sans relief explicite):

$$
\partial_t(M\mathbf u)+\nabla\!\cdot\!\Big(M\,\mathbf u\otimes\mathbf u+\tfrac12 gM^2\mathbf I\Big)
= Mf\,\hat k\times\mathbf u - Mr\,\mathbf u + \nabla\!\cdot(\nu M\nabla\mathbf u) + \mathbf F_{\rm oro}.
$$

* Énergie mécanique conservée si $r=\nu=\mathbf F_{\rm oro}=0$.

### 3.3 Eau atmosphérique (vapeur, nuage)

$$
\partial_t(Mq)+\nabla\!\cdot(Mq\mathbf u)=M\,(E-C+E_r)+\nabla\!\cdot(\kappa_q M\nabla q),
$$

$$
\partial_t(Mc)+\nabla\!\cdot(Mc\mathbf u)=M\,(C-P-\varepsilon c)+\nabla\!\cdot(\kappa_c M\nabla c).
$$

Paramétrisations minimalistes (stables):

$$
C=\frac{\max(0,\,q-q_{\rm sat})}{\tau_{\rm cond}},\quad
P=\frac{\max(0,\,c-c_{\rm crit})}{\tau_{\rm rain}},\quad
E_r=\min\!\Big(\tilde E_r,\ \frac{(q_{\rm sat}-q)_+}{\Delta t}\Big).
$$

Évaporation de surface dépendante du terrain:

$$
E=k_{\rm terr}\,\big(q_{\rm sat}(T,p_s)-q\big)_+\,\big(1+a_u\,|\mathbf u|\big).
$$

Bilan global de l’eau atmosphérique:

$$
\frac{d}{dt}\int M(q+c)\,dA = \int M\,(E-P+E_r-\varepsilon c)\,dA.
$$

### 3.4 Thermodynamique (température potentielle)

Définition: $\theta = T\,(p_0/p_s)^{R_d/c_p}$, avec $p_s=gM$.
Équation pronostique (pas d’adiabatique explicite):

$$
\boxed{\ C\,(\partial_t\theta+\mathbf u\!\cdot\nabla\theta)
= \frac{\theta}{T}\,\Big[Q_{\rm SW}^{\rm atm}-Q_{\rm LW}^{\rm atm}
+\kappa_T\nabla^2 T + Q_{\rm lat}^{\rm air} + Q_{\rm mech}\Big] \ }.
$$

Termes sources (W·m⁻²):

$$
Q_{\rm lat}^{\rm air}= L\,M\,(C - E_r),\qquad
Q_{\rm mech}\ \approx\ Mr\,|\mathbf u|^2 + \nu M\,\|\nabla \mathbf u\|^2 \ge 0.
$$

Récupération de $T$: $T=\theta\,(p_s/p_0)^{R_d/c_p}$.

> **Variante si on préfère $T$**: $C(\partial_t T+\mathbf u\!\cdot\nabla T)=Q_{\rm SW}^{\rm atm}-Q_{\rm LW}^{\rm atm}+\kappa_T\nabla^2 T - R_d T\,\nabla\!\cdot\mathbf u + Q_{\rm lat}^{\rm air}+Q_{\rm mech}$. Ne **pas** inclure $R_d T\,S_M/M$.

Compteur d’énergie de surface (sans $T_s$):

$$
\boxed{\ \frac{dE_{\rm sfc}}{dt} = -L\,M\,E\ +\ F_{\rm rad,sfc} + H_{\rm sens} + H_{\rm rain}\ }.
$$

---

## Rayonnement & astronomie

### 4.1 Géométrie solaire & orbite

Flux stellaire au TOA: $S_{\rm TOA}(t)=L_\star/(4\pi r(t)^2)$. Cosinus zénithal:

$$
\mu=\max\{0,\ \sin\phi\sin\delta+\cos\phi\cos\delta\cos h\},
$$

où $\delta$ est la déclinaison et $h$ l’angle horaire (déduits de l’orbite: $a,e,\varpi,P_{\rm orb}$).

### 4.2 Courte longueur d’onde (SW)

Air‑mass robuste (Kasten–Young):

$$
 m(\mu)=\frac{1}{\mu+0.50572\,(96.07995^\circ-\arccos\mu)^{-1.6364}}.
$$

Flux à la surface et absorption nette:

$$
S_{\rm surf}=S_{\rm TOA}\,\mu\,e^{-\tau_{\rm sw,loc}\,m(\mu)},\qquad
\boxed{\ Q_{\rm SW}=(1-\alpha)\,S_{\rm surf} \ }.
$$

Partage: $Q_{\rm SW}^{\rm atm}=\phi_{\rm SW}^{\rm atm} Q_{\rm SW}$, $Q_{\rm SW}^{\rm sfc}=\phi_{\rm SW}^{\rm sfc} Q_{\rm SW}$, avec $\phi_{\rm SW}^{\rm atm}+\phi_{\rm SW}^{\rm sfc}=1$.

### 4.3 Grande longueur d’onde (LW)

Schéma gris (2‑flux):

$$
\varepsilon_{\rm eff}=\frac{1}{1+\beta\,\tau_{\rm lw,loc}},\quad \beta\simeq \tfrac34,\qquad
\boxed{\ Q_{\rm LW}^{\rm atm}=\varepsilon_{\rm eff}\,\sigma T^4\ }.
$$

Opacité simple: $\tau_{\rm lw,loc}=k_{\rm lw}\,(p_s/g)\,q_{\rm abs}$.

---

## Saturation & thermodynamique humide

Clausius–Clapeyron générique (condensable quelconque):

$$
 e_s(T)=e_{s0}\,\exp\!\Big[\frac{L}{R_v}\Big(\frac{1}{T_0}-\frac{1}{T}\Big)\Big].
$$

Humidité de saturation (mélange condensable/gaz sec):

$$
 \boxed{\ q_{\rm sat}(T,p)=\frac{\varepsilon\,e_s(T)}{p-(1-\varepsilon)\,e_s(T)} },\qquad \varepsilon\equiv\frac{R_d}{R_v}=\frac{M_v}{M_d}.
$$

Humidité relative: $RH=q/q_{\rm sat}$. Température virtuelle (diagnostic dynamique):

$$
 T_v \approx T\,\big[1+(1/\varepsilon-1)\,q\big].
$$

---

## Pression à altitude fixe (hypsométrique)

Échelle de hauteur effective $H=R_d\,\overline{T_v}/g$. Pression à altitude géométrique $z$ (indépendante du relief):

$$
\boxed{\ p(z)=p_{\rm msl}\,e^{-z/H},\qquad p_{\rm msl}=gM\ }.
$$

Forme hypsométrique entre deux niveaux: $Z_2-Z_1=(R_d\overline{T_v}/g)\,\ln(p_1/p_2)$.

---

## Diagnostics & bilans

* **Énergie mécanique SW**: $\mathcal E_{\rm mech}=\tfrac12 M|\mathbf u|^2+\tfrac12 gM^2$.
* **Énergie totale (air + surface)**:

$$
\boxed{\ \mathcal E_{\rm tot}=\int\big[\tfrac12 M|\mathbf u|^2+\tfrac12 gM^2 + C T\big]\,dA\; + \; \int E_{\rm sfc}\,dA\ }.
$$

Sans rayonnement ni dissipation: $d\mathcal E_{\rm tot}/dt\approx 0$.

* **Pluie au sol**: $R=M\,P$ \[kg·m⁻²·s⁻¹] (1 mm·h⁻¹ ≈ 2.78e‑4 kg·m⁻²·s⁻¹).
* **PV barotrope (intuition)**: $q_{\rm PV}=(\zeta+f)/h$ avec $h=M/\rho_{\rm ref}$ (utile si orographie, voir ci‑dessous).

---

## Orographie conservative

Choisir $h=M/\rho_{\rm ref}$ et un fond fixe $b(\lambda,\phi)$ \[m]. Deux écritures équivalentes:

**(a) Flux de pression incluant le relief** (meilleure conservation d’énergie):

$$
\partial_t(h\mathbf u)+\nabla\!\cdot\!\Big(h\,\mathbf u\otimes\mathbf u + [\tfrac12 g h^2 + g h b] \mathbf I\Big)
= h f\,\hat k\times\mathbf u - h r\,\mathbf u + \nabla\!\cdot(\nu h\nabla\mathbf u).
$$

**(b) Force orographique explicite**:

$$
\partial_t(h\mathbf u)+\nabla\!\cdot\!(h\,\mathbf u\otimes\mathbf u + \tfrac12 g h^2 \mathbf I)
= -g\,h\,\nabla b + h f\,\hat k\times\mathbf u - h r\,\mathbf u + \nabla\!\cdot(\nu h\nabla\mathbf u).
$$

Potentiel d’énergie: $g(\tfrac12 h^2+h b)$.

---

## Paramètres planète‑agnostiques — pense‑bête

* **Gravité** $g$, **rotation** $\Omega$, **rayon** (pour aires), **gaz sec**: $M_d, R_d, c_p$, **condensable**: $M_v, R_v, L, e_{s0}, T_0$.
* **Rayonnement**: $L_\star$, éléments orbitaux $(a,e,\varpi,P_{\rm orb})$, $\alpha$, $\tau_{\rm sw,loc}$, $k_{\rm lw}, q_{\rm abs}$, $\beta\simeq3/4$.
* **Microphysique** (défauts raisonnables): $\tau_{\rm cond}\sim 600\,\mathrm{s}$, $c_{\rm crit}\sim 10^{-3}\!\text{–}\!5\times10^{-3}$, $\tau_{\rm rain}\sim 1\!\text{–}\!3\,\mathrm{h}$.
* **Surface**: hiérarchie $k_{\rm terr}$ (océan > forêt > sol nu > roche > neige/glace); option $a_u\in[0,0.2]$.

---

## Extensions possibles

* **Barocline 2 couches** (cisaillement vertical, instabilité barocline).
* **Surface active** ($T_s$ pronostique, océan/mix‑layer, réservoir d’eau).
* **Microphysique enrichie** (pluie explicite, glace, entrainement/détraînement).
* **Rayonnement non‑gris** (bandes spectrales, dépendances en humidité/CO₂).

---

### Notes pratiques

* Toujours évaluer $q_{\rm sat}(T,p)$ avec le $T$ et $p$ cohérents du pas courant.
* Compter le **latent** dans l’air via $C,E_r$ et à la **surface** via $E$ (pas de double comptage).
* Ne pas coupler la thermo au correcteur de masse $S_M$ (pas de terme $R_d T\,S_M/M$).