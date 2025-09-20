# Modèle atmosphérique barotrope (1 couche) — Doc Numérique / Implémentation

> **But** — Décrire des schémas numériques **conservatifs, stables et simples** pour intégrer les équations du document Physique sur une sphère.

---

## Table des matières

1. [Aperçu & principes](#aperçu--principes)
2. [Grille & géométrie sphérique](#grille--géométrie-sphérique)
3. [Disposition des variables (C‑grid)](#disposition-des-variables-cgrid)
4. [Flux-forme finies‑volumes (générique)](#flux-forme-finiesvolumes-générique)
5. [Advection masse & momentum (RK3‑TVD)](#advection-masse--momentum-rk3tvd)
6. [Advection des traceurs (limiteurs MUSCL)](#advection-des-traceurs-limiteurs-muscl)
7. [Orographie conservative (pression « flux‑pure »)](#orographie-conservative-pression--fluxpure-)
8. [Microphysique locale (sub‑pas)](#microphysique-locale-subpas)
9. [Rayonnement, surface & dissipation](#rayonnement-surface--dissipation)
10. [Diffusion & hyper‑viscosité](#diffusion--hyperviscosité)
11. [Contraintes CFL & choix du Δt](#contraintes-cfl--choix-du-Δt)
12. [Pôles & conditions aux limites](#pôles--conditions-aux-limites)
13. [Diagnostics & bilans (tests d’invariants)](#diagnostics--bilans-tests-dinvariants)
14. [Pseudo‑code d’un pas de temps complet](#pseudocode-dun-pas-de-temps-complet)
15. [Paramétrage recommandé (récap)](#paramétrage-recommandé-récap)

---

## Aperçu & principes

* **Toujours en flux‑forme finies‑volumes** : la conservation vient du télescopage des flux face‑à‑face.
* **Pondérer par l’aire** : intégrales et moyennes pondérées par $A_{i,j}$.
* **Séparation claire** : dynamique (advection) **puis** physiques locales (microphysique, rad, surface, diffusion).
* **Stabilité** : schémas centrés → limiter/filtrer, éviter upwind 1er ordre trop diffusif.

---

## Grille & géométrie sphérique

Soit une grille lat–lon régulière (rayon $a$). Pré‑calculs :

* Centres : $\lambda_i,\phi_j$. Demi‑niveaux : $\lambda_{i\pm1/2},\phi_{j\pm1/2}$.
* **Aire cellule** : $\boxed{A_{i,j}=a^2\,\Delta\lambda\,[\sin\phi_{j+1/2}-\sin\phi_{j-1/2}] }$.
* **Longueurs d’arêtes** :

  * Est/Ouest : $\boxed{L^{E/W}_{i\pm1/2,j}=a\,\Delta\phi}$
  * Nord/Sud : $\boxed{L^{N/S}_{i,j\pm1/2}=a\,\cos\phi_{j\pm1/2}\,\Delta\lambda}$
* Pas métriques (diagnostics) : $\Delta x=a\cos\phi_j\,\Delta\lambda,\ \Delta y=a\,\Delta\phi$.
* **Intégrales globales** : $\sum_{i,j}\Psi_{i,j}A_{i,j}$.

> Alternatives sans pôles : cubed‑sphere / icosaédrique (mêmes recettes finies‑volumes avec leurs aires/arêtes).

---

## Disposition des variables (C‑grid)

* **Centres** (cell centers) : $M, q, c, \theta$ (ou $T$), $b$ (relief), $h=M/\rho_{\rm ref}$.
* **Arêtes Est/Ouest** : $u$ (zonal), **Arêtes Nord/Sud** : $v$ (méridien).
* Les flux massiques naturels sont $MU$ sur E/W et $MV$ sur N/S.

---

## Flux-forme finies‑volumes (générique)

Pour une cellule $(i,j)$, et une quantité surfacique $\Psi$ au centre (ex. $M$, $Mq$) :

**Flux aux faces** (ex. face Est) : $F^E = (\text{valeur face}) \times L^E$.
**Divergence** :
$\boxed{\ (\nabla\!\cdot \mathbf F)_{i,j}=\frac{F^E_{i+1/2,j}-F^E_{i-1/2,j}+F^N_{i,j+1/2}-F^S_{i,j-1/2}}{A_{i,j}} \ }$

> **Clé** : construire les **valeurs de face** par interpolation **upwind limitée** (MUSCL/Van Leer/MC), pour stabilité + faible diffusion.

---

## Advection masse & momentum (RK3‑TVD)

On note $Y\equiv (M, MU, MV)$ et $\mathcal{L}(Y)$ les tendances (divergence de flux + sources).
**RK3 (Shu–Osher) :**

```
Y^1 = Y^n + Δt · L(Y^n)
Y^2 = 3/4 · Y^n + 1/4 · [ Y^1 + Δt · L(Y^1) ]
Y^{n+1} = 1/3 · Y^n + 2/3 · [ Y^2 + Δt · L(Y^2) ]
```

**Construction des flux :**

* **Masse** :

  * Interpoler M aux faces → $M_{i+1/2,j}^{\uparrow}$ (upwind limité).
  * Flux E/W : $F_M^E = (M U)_{i+1/2,j} · L^E$, avec $U$ déjà défini sur l’arête.
* **Momentum** :

  * Advecter $MU$ et $MV$ comme des quantités conservatives (mêmes flux‑forme).
  * **Pression barotrope** (sans relief) : ajouter le flux isotrope $+ (\tfrac12 g M^2) I$ (voir orographie ci‑dessous pour la version complète).
* **Coriolis** : terme source local (conservatif en énergie au continu ; au discret, l’appliquer centré en temps quand possible).
* **Friction linéaire** : source $-M r u$ / $-M r v$ (voir dissipation → chaleur).

> Astuce : calculer d’abord des **flux de masse** $F_M$ et réutiliser leurs vitesses « de face » pour tous les scalaires.

---

## Advection des traceurs (limiteurs MUSCL)

Pour un traceur conservatif $\Phi = M q$ :

1. Reconstruire $q$ (ou $\Phi$) sur chaque face avec un **limiteur** (Van Leer, MC) :

   * pente limitée $\sigma$ à partir des valeurs voisines,
   * états gauche/droite $q_L, q_R$ à la face.
2. Choisir l’état **amont** avec la vitesse de face (signe de $U$ ou $V$).
3. Construire le **flux de $\Phi$** : $F_\Phi^E = (M U)^E · q^{\text{amont}} · L^E$.
4. Divergence sur la cellule (formule générique ci‑dessus).
5. **Positivité** : clipper $q, c \ge 0$ après mise à jour ; limiter le flux pour ne pas vider une cellule en un pas.

> Même recette pour $Mc, M\theta$ (si vous advectez $\theta$ en forme conservative via $C$ constant).

---

## Orographie conservative (pression « flux‑pure »)

Travail en hauteur $h=M/\rho_{ref}$, fond fixe $b(\lambda,\phi)$, surface libre $\eta=h+b$.

**Flux de pression complet (meilleur bilan d’énergie)** :
$\Phi_p = g(\tfrac12 h^2 + h b)$
Ajoute au tenseur de flux momentum :
$\mathbf{F}^{(p)} = \Phi_p · \mathbf I$
Discrétisation : à chaque face $f$ d’aire linéique $L_f$, utiliser $\Phi_{p,f} = g(\tfrac12 h_f^2 + h_f b_f)$ avec **moyennes centrées** (ou limitées) $h_f, b_f$, puis ajouter
$F^{(p)}_f = \Phi_{p,f} · L_f$
à la somme des flux de momentum (E/W/N/S).

**Équivalent « force »** (si préféré) : ajouter $-g h\,\nabla b$ comme source ; moins propre pour l’énergie au discret.

> **Lissage doux** de $b$ (1–2 mailles) recommandé pour éviter les « PGF errors » si le relief est bruité.

---

## Microphysique locale (sub‑pas)

But : résoudre les ODE locales raides pour $q, c$ sans déstabiliser l’advection.

**Schéma** (dans chaque cellule, après advection) :

```
// choisir nombre de sous-pas
τ_min = min(τ_cond, τ_rain)
N = max(1, ceil(Δt / (f_sub * τ_min)))     // f_sub ≈ 0.8–1.0
δt = Δt / N
repeat N times:
  qsat = q_sat(T, p_s)              // cohérent avec le T actuel
  C = max(0, (q - qsat)) / τ_cond
  P = max(0, (c - c_crit)) / τ_rain
  Er = min(Ēr_pred, max(0, qsat - q) / δt)  // borné par undersaturation
  // limiter aux stocks disponibles
  C = min(C, q / δt)
  P = min(P, c / δt)
  // mises à jour conservatives (formes surfaciques)
  q -= δt * (C - Er)
  c += δt * (C - P - ε c)
end
```

**Chaleur latente** pour l’air : ajouter $+L·M·(C−Er)$ à $Q_{lat}^{air}$.
**Surface** : stocker $−L·M·E$ dans $E_{sfc}$ lors de l’étape surface (cf. plus bas).

---

## Rayonnement, surface & dissipation

Étapes **locales** (pas de flux) appliquées après la microphysique :

* **Rayonnement** :

  * SW : calculer $Q_{SW}$ via Beer–Lambert + géométrie ; répartir ($φ_{SW}^{atm}, φ_{SW}^{sfc}$).
  * LW : $Q_{LW}^{atm}=\varepsilon_{eff} σ T^4$.
  * Mise à jour de $\theta$ :
    $C · Δ\theta = (\theta/T) · (Q_{SW}^{atm} - Q_{LW}^{atm} + \kappa_T∇^2T + Q_{lat}^{air} + Q_{mech}) · Δt$
* **Surface** : compteur d’énergie
  $ΔE_{sfc} = (-L·M·E + F_{rad,sfc} + H_{sens} + H_{rain}) · Δt$
* **Dissipation mécanique** (drag/viscosité) → chaleur : inclure $Q_{mech}$ ci‑dessus (optionnel au début : ignorer si petit).

---

## Diffusion & hyper‑viscosité

**Option simple (Laplacien)** en flux‑forme :

* Flux diffusif d’un scalaire $\psi$ : $F^{diff}_f = -\kappa · M_f · (\nabla \psi \cdot n)_f · L_f$.
* Divergence → tendance. Choisir $\kappa$ pour stabiliser sans sur‑diffuser.

**Option recommandée (hyper‑viscosité)** : appliquer $−ν_4 ∇^4$ (ou $−ν_6 ∇^6$) sur $M, u, q, c, θ$ via filtrage spectral simple (si dispo) ou composition de Laplaciens. Caler $ν_4$ pour ne dissiper que aux dernières 3–5 mailles.

---

## Contraintes CFL & choix du Δt

* **Advectif** : $C_u = |u| Δt / Δx$, $C_v = |v| Δt / Δy$ → $\max(C_u,C_v) ≤ C_{max}$ (typiquement 0.5–0.8 selon schéma).
* **Ondes de gravité SW** : vitesse $c_g = √(g h_{eff})$ (prendre $h_{eff}$ médian/glob.). Exiger $c_g Δt / \min(Δx,Δy) ≤ C_g$ (≈ 0.5).
* **Diffusion** (explicite) : $Δt ≤ (Δx^2)/(4ν)$ 2D.
* **Pôles (lat–lon)** : $Δx ∝ cosφ$ → réduire $Δt$ en conséquence (ou filtrer).

---

## Pôles & conditions aux limites

* **Longitude** : périodique (copie fantômes i=0 ↔ i=N\_λ).
* **Pôles** (lat–lon) :

  * Cap polaire : regrouper plusieurs longitudes en un nœud (super‑cellule), ou
  * Filtre zonal léger $|φ|>φ_{cap}$ (p.ex. 85°), ou
  * Grille alternative (cubed‑sphere) à terme.
* **Relief** : aucun masquage (l’atmosphère recouvre tout) ; le relief n’entre qu’en pression.

---

## Diagnostics & bilans (tests d’invariants)

À chaque pas (ou tous N pas), calculer et logger :

* **Masse** : $\sum M A$.
* **Eau** : $\sum M(q+c) A$ et le RHS $\sum M(E−P+E_r−εc) A$.
* **Énergie** : $\sum [½ M|u|^2 + ½ g M^2 + C T] A + \sum E_{sfc} A$.
* **Courant CFL** max advectif & gravité.
* **PV barotrope** (si orographie) : $q_{PV}=(ζ+f)/h$, statistique globale.
* **Spectre cinétique** (optionnel) : surveiller pente et énergie aux plus petites échelles.

> **Critères d’alerte** : q/c négatifs, dérive d’énergie sans forçages, oscillations peigne, instabilité polaire.

---

## Pseudo‑code d’un pas de temps complet

```
// Entrées au pas n : M, u, v, q, c, θ (ou T), E_sfc, b, params_planète
// 1) Géométrie & pré-calculs (faite 1x au setup) : A, L_E/W/N/S, métriques, masques périodiques

// 2) RK3 sur (M, MU, MV)
for stage s in {1,2,3}:
  // 2.1 Flux masse
  MU = U_on_faces(u) * M_faces(upwind-limited)
  MV = V_on_faces(v) * M_faces(upwind-limited)
  F_M = {E/W/N/S} = (MU, MV) * L_{faces}
  divM = divergence(F_M) / A

  // 2.2 Flux momentum (advectif + pression + diff)
  F_MU_adv = advect(MU)   // même logique que masse
  F_MV_adv = advect(MV)
  Φ_p_face = g * (0.5*h_face^2 + h_face * b_face)
  F_p = Φ_p_face * I * L_{faces}
  divMU = divergence(F_MU_adv + F_p_x + F_diff_u) / A
  divMV = divergence(F_MV_adv + F_p_y + F_diff_v) / A

  // 2.3 Sources dynamiques
  S_M   = mass_fixer(M)
  S_MU  = coriolis(M,u,v) - M*r*u
  S_MV  = coriolis(M,u,v) - M*r*v

  // 2.4 Tendances
  L_M  = -divM + S_M
  L_MU = -divMU + S_MU
  L_MV = -divMV + S_MV

  // 2.5 Mise à jour RK3
  (M, MU, MV) = rk3_update((M,MU,MV), (L_M,L_MU,L_MV), s, Δt)
end
u = MU / M_on_u ; v = MV / M_on_v          // re-projeter sur arêtes

// 3) Advection des traceurs (q, c, θ)
for tracer in {Mq, Mc, Mθ}:   // ou M*T si vous restez en T
  F_tr = flux_conservative(M, tracer, u, v, limiter)
  tracer += -Δt * divergence(F_tr)/A
// clip q,c ≥ 0 ; limiter pour éviter de vider une cellule

// 4) Microphysique locale (sub-pas)
(q, c, Q_lat_air) = microphysics_step(q, c, T, p_s, Δt)

// 5) Surface & rayonnement
Q_SW, Q_LW = radiative_fluxes(T, ...)
θ += (θ/T) * (Q_SW_atm - Q_LW_atm + κ_T ∇²T + Q_lat_air + Q_mech) * Δt / C
E_sfc += (-L*M*E + F_rad_sfc + H_sens + H_rain) * Δt
T = θ * (p_s/p0)^(R_d/c_p)   // si variable pronostique = θ

// 6) Diffusion/hyperviscosité (si séparée)
apply_diffusion(M, u, v, q, c, θ)

// 7) Pôles (lat-lon) : filtre léger si nécessaire
polar_filter(u, v, q, c, θ)

// 8) Diagnostics & logs
compute_and_log_budgets(M, u, v, q, c, θ, E_sfc, A)
```

---

## Paramétrage recommandé (récap)

* **Time stepping** : RK3‑TVD ; Δt tel que CFL\_adv ≲ 0.6 ; CFL\_grav ≲ 0.5.
* **Limiteurs** : Van Leer ou MC pour q,c,θ.
* **Microphysique** : τ\_cond ≈ 600 s ; c\_crit ∈ \[1e−3, 5e−3] ; τ\_rain ≈ 1–3 h ; sub‑pas dès Δt > 0.8·τ\_cond.
* **Orographie** : forme flux‑pure (Φ\_p = g(½h² + hb)); lissage b(x,y) σ ≈ 1–2 mailles si bruitée.
* **Diffusion** : hyper‑ν réglé pour dissiper aux 3–5 dernières mailles ; Laplacien minimal sinon.
* **Pôles** : cap/filtres au‑delà de 85° si lat–lon.

---

### Notes d’implémentation

* **Unités** : tracer des asserts d’unités en debug (Pa, W·m⁻², kg·m⁻²·s⁻¹, etc.).
* **Reproductibilité** : logger Δt, schémas, constantes, seeds ; sauvegarder les bilans.
* **Vectorisation** : privilégier des opérations par faces (tableaux 2D) pour perf.
* **Tests** : mettre en place 3 cas automatiques (onde de Rossby forcée, bump orographique, patch humide advecté) qui valident les invariants et la stabilité.