# Documentation Développeur Nexus-Cosmic

**Version:** 1.0.0  
**Langue:** Français | [English](DOCUMENTATION_EN.md)

---

## Table des Matières

1. [Démarrage Rapide](#démarrage-rapide)
2. [Concepts Fondamentaux](#concepts-fondamentaux)
3. [Référence API](#référence-api)
4. [Patterns & Cas d'Usage](#patterns--cas-dusage)
5. [Lois de Force Expliquées](#lois-de-force-expliquées)
6. [Topologies Expliquées](#topologies-expliquées)
7. [Mécanisme de Gel](#mécanisme-de-gel)
8. [Extensibilité](#extensibilité)
9. [Optimisation Performance](#optimisation-performance)
10. [Dépannage](#dépannage)

---

## Démarrage Rapide

### Installation

```bash
pip install git+https://github.com/Tryboy869/nexus-cosmic.git
```

### Utilisation Basique

```python
from nexus_cosmic import NexusCosmic

# Consensus distribué
system = NexusCosmic(mode='consensus', n_entities=100)
result = system.run()
print(f"Consensus: {system.get_consensus()}")
print(f"Étapes: {result['steps']}")

# Tri émergent
system = NexusCosmic(mode='sorting', values=[8,3,9,1,5])
system.run()
print(system.get_sorted_values())  # [1,3,5,8,9]
```

---

## Concepts Fondamentaux

### Qu'est-ce que Nexus-Cosmic ?

Nexus-Cosmic est un **framework de calcul distribué** basé sur les **principes physiques émergents**. Au lieu d'algorithmes centralisés, il utilise des interactions locales simples entre entités qui convergent naturellement vers des solutions optimales.

### Principes Clés

1. **Émergence**: Comportement global complexe à partir de règles locales simples
2. **Réseaux Petit-Monde**: Propagation d'information en O(log N)
3. **Momentum**: Inertie inspirée de la physique pour la stabilité
4. **Mécanisme de Gel**: Économie computationnelle (100% en régime stable)
5. **Attracteurs Discrets**: Convergence garantie pour le tri

### Architecture

```
Système NexusCosmic
├── Entités (UniversalEntity)
│   ├── state (position dans l'espace de solution)
│   ├── velocity (momentum)
│   ├── mass (inertie)
│   └── frozen (état de gel)
├── Topologie (connexions voisins)
│   ├── small_world
│   ├── full
│   ├── ring
│   └── grid
├── Loi de Force (règles d'interaction)
│   ├── adaptive_consensus
│   └── discrete_attractor
└── Mécanisme de Gel (économie)
    ├── threshold
    └── stability_steps
```

---

## Référence API

### Classe NexusCosmic

#### Constructeur

```python
NexusCosmic(
    mode=None,                    # 'consensus', 'sorting', ou None (custom)
    n_entities=None,              # Nombre d'entités (requis pour consensus/custom)
    values=None,                  # Valeurs à trier (requis pour sorting)
    topology='small_world',       # Topologie réseau
    shortcuts_per_node=2,         # Pour topologie small_world
    momentum=0.8,                 # Facteur momentum [0, 1]
    strength=None,                # Force (auto si None)
    force_law=None,               # Loi de force custom
    freeze_enabled=True,          # Activer mécanisme gel
    freeze_threshold=0.01,        # Seuil stabilité
    freeze_stability_steps=5,     # Étapes avant gel
    seed=None                     # Seed aléatoire
)
```

#### Paramètres Expliqués

**mode** (str): Mode opératoire
- `'consensus'`: Consensus distribué (toutes entités convergent vers même valeur)
- `'sorting'`: Tri émergent (entités s'organisent par valeur)
- `None`: Mode custom (définir votre propre loi)

**n_entities** (int): Nombre d'entités dans le système
- Consensus: 10-1000 (performance dégrade >1000)
- Custom: N'importe quel entier positif

**values** (list): Valeurs pour mode sorting
- Doivent être numériques (int ou float)
- N'importe quelle longueur (testé jusqu'à 10,000)

**topology** (str): Structure réseau
- `'small_world'`: Meilleure performance (diamètre O(log N))
- `'full'`: Plus rapide pour petit N (<50)
- `'ring'`: Expérimental (lent mais fiable)
- `'grid'`: Expérimental (grille 2D)

**momentum** (float): Facteur d'inertie [0, 1]
- `0.0`: Pas de momentum (réactif)
- `0.8`: Recommandé (équilibré)
- `1.0`: Inertie maximale (lent à changer)

**strength** (float ou callable): Magnitude force
- `None`: S'adapte auto à N (recommandé)
- `float`: Force fixe
- `callable`: Fonction de N, ex: `lambda n: 0.1 * log(n)`

**freeze_threshold** (float): Seuil variance pour gel
- `0.01`: Recommandé (convergence serrée)
- `0.1`: Plus lâche (plus rapide mais moins précis)

**freeze_stability_steps** (int): Étapes stables avant gel
- `5`: Recommandé (fiable)
- `1`: Agressif (risque oscillations)
- `10`: Conservateur (gel plus lent)

#### Méthodes

##### run()

Exécute simulation jusqu'à convergence.

```python
result = system.run(
    max_steps=100,              # Itérations maximum
    convergence_threshold=1.0,  # Seuil variance
    verbose=False               # Afficher progrès
)
```

**Retourne:** dict avec clés:
- `converged` (bool): Si système a convergé
- `steps` (int): Nombre d'étapes effectuées
- `final_variance` (float): Variance finale
- `final_freeze_ratio` (float): Ratio entités gelées [0, 1]
- `active_entities` (int): Nombre entités actives

##### get_consensus()

Obtenir valeur consensus (moyenne états toutes entités).

```python
consensus = system.get_consensus()  # float
```

##### get_sorted_values()

Obtenir valeurs triées (mode sorting).

```python
sorted_vals = system.get_sorted_values()  # list
```

##### inject_change()

Injecter changement local (dégèle zone affectée).

```python
system.inject_change(
    entity_id=15,      # Entité à modifier
    new_state=100.0    # Nouvelle valeur état
)
```

**Cas d'usage:** Tester résilience, simuler événements externes

##### reset()

Réinitialiser système à état initial.

```python
system.reset()
```

##### variance()

Calculer variance actuelle.

```python
var = system.variance()  # float
```

##### count_active()

Compter entités actives (non-gelées).

```python
active = system.count_active()  # int
```

##### get_frozen_ratio()

Obtenir ratio entités gelées.

```python
ratio = system.get_frozen_ratio()  # float [0, 1]
```

---

## Patterns & Cas d'Usage

### Pattern 1: Consensus Distribué

**Problème:** N nœuds doivent s'accorder sur une valeur sans coordinateur central.

**Solution:**

```python
from nexus_cosmic import NexusCosmic

# Initialiser avec états aléatoires
system = NexusCosmic(mode='consensus', n_entities=100)

# Exécuter jusqu'à convergence
result = system.run()

# Obtenir valeur convenue
consensus = system.get_consensus()

print(f"Les 100 nœuds s'accordent sur: {consensus}")
print(f"Convergé en {result['steps']} étapes")
```

**Applications réelles:**
- Protocoles consensus blockchain
- Synchronisation cache distribué
- Coordination multi-agents
- Fusion réseau capteurs

### Pattern 2: Tri Émergent

**Problème:** Trier valeurs avec entités distribuées.

**Solution:**

```python
from nexus_cosmic import NexusCosmic

# Données non triées
values = [42, 17, 91, 3, 58, 24, 76]

# Créer système tri
system = NexusCosmic(mode='sorting', values=values)

# Exécuter tri
system.run()

# Obtenir résultat trié
sorted_values = system.get_sorted_values()

print(f"Trié: {sorted_values}")
# Sortie: [3, 17, 24, 42, 58, 76, 91]
```

**Applications réelles:**
- Files priorité dynamiques
- Ordonnancement tâches
- Systèmes classement
- Allocation ressources

### Pattern 3: Loi de Force Custom

**Problème:** Besoin règles interaction personnalisées.

**Solution:**

```python
from nexus_cosmic import NexusCosmic, CustomLaw
import math

class LoiGravite(CustomLaw):
    """Entités attirent proportionnellement à différence masse"""
    
    def compute_force(self, entity1, entity2):
        # Magnitude force
        distance = abs(entity2.state - entity1.state)
        if distance < 0.01:
            return 0.0
        
        force = (entity2.mass * entity1.mass) / (distance ** 2)
        
        # Direction
        direction = 1 if entity2.state > entity1.state else -1
        
        return force * direction * 0.01

# Utiliser loi custom
system = NexusCosmic(
    n_entities=50,
    force_law=LoiGravite(),
    topology='small_world'
)

result = system.run()
```

**Applications réelles:**
- Simulations particules
- Problèmes optimisation
- IA jeux (flocking, essaimage)
- Routage réseau

### Pattern 4: Mises à Jour Dynamiques

**Problème:** Système doit s'adapter aux changements.

**Solution:**

```python
from nexus_cosmic import NexusCosmic

# Système initial
system = NexusCosmic(mode='consensus', n_entities=50)
system.run()

consensus_initial = system.get_consensus()
print(f"Consensus initial: {consensus_initial}")

# Injecter changement externe
system.inject_change(entity_id=25, new_state=100.0)

# Système se ré-adapte
system.run()

nouveau_consensus = system.get_consensus()
print(f"Nouveau consensus: {nouveau_consensus}")
```

**Applications réelles:**
- Systèmes temps réel
- Réseaux adaptatifs
- Tolérance pannes
- Systèmes auto-réparation

### Pattern 5: Calcul Hardware Faible

**Problème:** Ressources computationnelles limitées.

**Solution:**

```python
from nexus_cosmic import NexusCosmic
import random

class MoteurHardwareFaible:
    def __init__(self, n_workers=100):
        self.system = NexusCosmic(
            mode='consensus',
            n_entities=n_workers,
            freeze_enabled=True  # Critique pour économie
        )
    
    def moyenne_distribuee(self, large_dataset):
        # Chaque worker traite petit morceau
        chunk_size = len(large_dataset) // self.system.n
        
        for i, entity in enumerate(self.system.entities):
            start = i * chunk_size
            end = start + chunk_size
            chunk = large_dataset[start:end]
            entity.state = sum(chunk) / len(chunk)
        
        # Moyenne globale émergente
        self.system.run()
        
        return self.system.get_consensus()

# Utiliser sur appareil mobile
moteur = MoteurHardwareFaible(n_workers=100)
big_data = [random.random() for _ in range(1_000_000)]

# Chaque worker: 10K valeurs seulement
avg = moteur.moyenne_distribuee(big_data)
```

**Applications réelles:**
- Calcul distribué mobile
- Edge computing IoT
- Apprentissage fédéré
- Environnements ressources limitées

---

## Lois de Force Expliquées

### Loi Consensus Adaptatif

**But:** Faire converger entités vers valeur moyenne.

**Formule:**
```
force = (état_voisin - état_entité) × strength
```

**Comportement:**
- Entités attirent vers voisins
- Strength s'adapte à taille système N
- Garantit convergence

**Paramètres:**
- `strength`: Auto ou `0.1 × log(N) / log(10)`

**Utiliser quand:**
- Besoin accord distribué
- Vouloir valeur moyenne/médiane
- Nécessiter tolérance pannes

**Code:**
```python
from nexus_cosmic.core.laws import ForceLaw

loi = ForceLaw.adaptive_consensus(strength=0.2)
```

### Loi Attracteur Discret

**But:** Trier entités en créant positions discrètes.

**Formule:**
```
position_cible = rang_entité
force = (position_cible - position_actuelle) × strength
```

**Comportement:**
- Chaque entité attirée vers position rang
- Crée ordre trié
- Attracteurs discrets empêchent chevauchement

**Paramètres:**
- `strength`: `0.3` (fixe)
- `method`: `'discrete_attractors'`

**Utiliser quand:**
- Besoin sortie triée
- Vouloir ordre garanti
- Systèmes basés priorités

**Code:**
```python
from nexus_cosmic.core.laws import ForceLaw

loi = ForceLaw.discrete_attractor()
```

### Lois Custom

Créez vos propres règles interaction.

**Template:**

```python
from nexus_cosmic import CustomLaw

class MaLoi(CustomLaw):
    def __init__(self, param1=1.0):
        self.param1 = param1
    
    def compute_force(self, entity1, entity2):
        """
        Calculer force sur entity1 de entity2
        
        Args:
            entity1: Entité recevant force
            entity2: Entité exerçant force
        
        Returns:
            float: Magnitude force (positif = tire droite, négatif = tire gauche)
        """
        # Votre logique ici
        diff = entity2.state - entity1.state
        force = diff * self.param1
        return force
```

**Exemples:**

#### Décroissance Exponentielle

```python
class LoiExponentielle(CustomLaw):
    def __init__(self, decay=0.5):
        self.decay = decay
    
    def compute_force(self, e1, e2):
        import math
        distance = abs(e2.state - e1.state)
        magnitude = math.exp(-self.decay * distance)
        direction = 1 if e2.state > e1.state else -1
        return magnitude * direction * 0.1
```

#### Force Ressort

```python
class LoiRessort(CustomLaw):
    def __init__(self, k=0.1, amortissement=0.9):
        self.k = k
        self.amortissement = amortissement
    
    def compute_force(self, e1, e2):
        # Loi Hooke avec amortissement
        deplacement = e2.state - e1.state
        force_ressort = self.k * deplacement
        force_amortissement = -self.amortissement * e1.velocity
        return force_ressort + force_amortissement
```

#### Activation Seuil

```python
class LoiSeuil(CustomLaw):
    def __init__(self, seuil=1.0, strength=0.2):
        self.seuil = seuil
        self.strength = strength
    
    def compute_force(self, e1, e2):
        diff = e2.state - e1.state
        if abs(diff) < self.seuil:
            return 0.0  # Pas de force si dans seuil
        return diff * self.strength
```

---

## Topologies Expliquées

### Topologie Petit-Monde

**Structure:** Clusters locaux + raccourcis longue portée

**Diamètre:** O(log N)

**Performance:** Meilleure (speedup 27x)

**Visuel:**
```
Entité 0: [1, 2, 15, 42]  (voisins: locaux + 2 aléatoires)
Entité 1: [0, 2, 3, 28]
Entité 2: [0, 1, 3, 7]
...
```

**Utiliser quand:**
- Choix par défaut (presque toujours)
- Besoin convergence rapide
- Avoir >20 entités

**Code:**
```python
system = NexusCosmic(
    n_entities=100,
    topology='small_world',
    shortcuts_per_node=2  # Nombre raccourcis aléatoires
)
```

### Connectivité Complète

**Structure:** Chaque entité connectée à toutes les autres

**Diamètre:** 1

**Performance:** Meilleure pour N < 50, coûteuse pour grand N

**Visuel:**
```
Entité 0: [1, 2, 3, 4, ..., 99]  (toutes autres)
Entité 1: [0, 2, 3, 4, ..., 99]
...
```

**Utiliser quand:**
- Très petits systèmes (N < 50)
- Besoin convergence absolument la plus rapide
- Pas de souci scalabilité

**Code:**
```python
system = NexusCosmic(
    n_entities=30,
    topology='full'
)
```

### Topologie Anneau

**Structure:** Chaque entité connectée à 2 voisins (circulaire)

**Diamètre:** N/2

**Performance:** Lente mais fiable (expérimental)

**Visuel:**
```
Entité 0: [99, 1]  (précédent, suivant)
Entité 1: [0, 2]
Entité 2: [1, 3]
...
Entité 99: [98, 0]
```

**Utiliser quand:**
- Tests/recherche
- Vouloir structure prévisible
- Besoin isolation pannes

**Code:**
```python
system = NexusCosmic(
    n_entities=100,
    topology='ring'
)
```

### Topologie Grille

**Structure:** Grille 2D (4 voisins sauf bords)

**Diamètre:** 2×sqrt(N)

**Performance:** Lente mais stable (expérimental)

**Visuel:**
```
Grille 5×5:
0  1  2  3  4
5  6  7  8  9
10 11 12 13 14
15 16 17 18 19
20 21 22 23 24

Entité 12: [7, 11, 13, 17]  (haut, gauche, droite, bas)
Entité 0: [1, 5]            (droite, bas seulement)
```

**Utiliser quand:**
- Problèmes spatiaux
- Traitement image
- Automates cellulaires

**Code:**
```python
system = NexusCosmic(
    n_entities=25,  # Créera grille 5×5
    topology='grid'
)
```

---

## Mécanisme de Gel

### Qu'est-ce que le Gel ?

Quand l'état d'une entité devient stable (variance sous seuil pendant N étapes), elle **gèle** et arrête de calculer. Cela atteint **100% d'économie computationnelle** en régime stable.

### Comment ça Marche

```python
class UniversalEntity:
    def check_stability(self, threshold=0.01, stability_steps=5):
        # Comparer à état précédent
        change = abs(self.state - self.previous_state)
        
        if change < threshold:
            self.stability_counter += 1
            if self.stability_counter >= stability_steps:
                self.frozen = True  # Gèle!
        else:
            self.stability_counter = 0
```

### Configuration

```python
system = NexusCosmic(
    mode='consensus',
    n_entities=100,
    freeze_enabled=True,          # Activer gel
    freeze_threshold=0.01,        # Seuil stabilité
    freeze_stability_steps=5      # Étapes stables avant gel
)
```

### Surveiller Gel

```python
# Pendant simulation
result = system.run()
print(f"Gelé: {result['final_freeze_ratio']*100:.0f}%")
print(f"Actif: {result['active_entities']}")

# Obtenir ratio gel
ratio = system.get_frozen_ratio()  # float [0, 1]

# Compter actifs
active = system.count_active()  # int
```

### Dégel

Entités dégèlent automatiquement quand:
1. Elles reçoivent changement externe (inject_change)
2. Leurs voisins changent significativement
3. Système est réinitialisé

```python
# Injecter changement (dégèle zone)
system.inject_change(entity_id=50, new_state=100.0)

# Vérifier dégelé
active_apres = system.count_active()  # Plus élevé qu'avant
```

### Bénéfices

1. **Économie Computationnelle**: 100% économies en régime stable
2. **Efficacité Énergétique**: Critique pour IoT/mobile
3. **Scalabilité**: Grands systèmes s'auto-optimisent
4. **Tolérance Pannes**: Changements locaux affectent seulement zone locale

---

## Extensibilité

### Topologies Custom

```python
from nexus_cosmic import CustomTopology

class TopologieHexagonale(CustomTopology):
    """Grille hexagonale (6 voisins)"""
    
    def get_neighbors(self, entity_id, n_entities):
        """
        Retourne liste IDs voisins
        
        Args:
            entity_id: ID entité courante
            n_entities: Nombre total entités
        
        Returns:
            list of int: IDs voisins
        """
        # Votre logique topologie
        neighbors = []
        
        # Exemple: voisins hexagonaux
        row_size = int(n_entities ** 0.5)
        row = entity_id // row_size
        col = entity_id % row_size
        
        # Ajouter 6 voisins (pattern hex)
        # ... (implémentation)
        
        return neighbors

# Utiliser
system = NexusCosmic(
    n_entities=100,
    topology=TopologieHexagonale()
)
```

### Conditions Gel Custom

```python
from nexus_cosmic import CustomFreeze

class GelEnergie(CustomFreeze):
    """Geler basé sur niveau énergie"""
    
    def __init__(self, seuil_energie=0.1):
        self.seuil = seuil_energie
    
    def should_freeze(self, entity):
        """
        Vérifier si entité doit geler
        
        Args:
            entity: Instance UniversalEntity
        
        Returns:
            bool: True si doit geler
        """
        # Calculer énergie cinétique
        energie = 0.5 * entity.mass * (entity.velocity ** 2)
        
        return energie < self.seuil

# Utiliser
from nexus_cosmic.core.engine import NexusCosmic

system = NexusCosmic(n_entities=50)
system.freeze_condition = GelEnergie(seuil_energie=0.05)
```

### Outil Validation

Testez vos lois custom avant déploiement.

```python
from nexus_cosmic import validate_law, CustomLaw

class MaLoi(CustomLaw):
    def compute_force(self, e1, e2):
        return (e2.state - e1.state) * 0.15

# Valider
results = validate_law(
    MaLoi(),
    n_runs=5,         # Nombre tests
    max_steps=100,    # Max étapes par test
    n_entities=30     # Taille système
)

print(f"Taux succès: {results['success_rate']}%")
print(f"Étapes moy: {results['avg_steps']}")
print(f"Verdict: {results['verdict']}")
# Sortie: ✅ VALIDÉ, ⚠️ EXPERIMENTAL, ou ❌ REJETÉ
```

---

## Optimisation Performance

### Accélération NumPy (Optionnel)

```bash
pip install numpy
```

Nexus-Cosmic détecte et utilise automatiquement NumPy pour **speedup 2-3x** sur grands systèmes.

### Choisir Paramètres Optimaux

**Pour vitesse:**
```python
system = NexusCosmic(
    mode='consensus',
    n_entities=100,
    topology='small_world',    # Meilleure topologie
    shortcuts_per_node=3,      # Plus raccourcis = plus rapide
    momentum=0.9,              # Momentum élevé = stabilité
    freeze_threshold=0.1       # Plus lâche = gel plus rapide
)
```

**Pour précision:**
```python
system = NexusCosmic(
    mode='consensus',
    n_entities=100,
    topology='small_world',
    shortcuts_per_node=2,
    momentum=0.7,              # Plus bas = plus réactif
    freeze_threshold=0.001,    # Plus serré = plus précis
    freeze_stability_steps=10  # Plus étapes = fiable
)
```

### Scalabilité

| N Entités | Topologie | Étapes Moy | Temps (ms) |
|-----------|-----------|------------|------------|
| 10        | full      | 8          | 0.5        |
| 50        | small_world | 12       | 2.1        |
| 100       | small_world | 15       | 5.3        |
| 500       | small_world | 22       | 45         |
| 1000      | small_world | 28       | 180        |

**Recommandation:** Utiliser `small_world` pour N > 20

---

## Dépannage

### Système ne converge pas

**Symptôme:** `result['converged'] == False`

**Solutions:**
1. Augmenter `max_steps`
2. Vérifier loi force (utiliser `validate_law`)
3. Essayer topologie différente
4. Ajuster paramètre `strength`

```python
# Debug
result = system.run(verbose=True)  # Voir étape par étape
```

### Convergence lente

**Symptôme:** Prend >100 étapes

**Solutions:**
1. Utiliser `topology='small_world'`
2. Augmenter `shortcuts_per_node`
3. Ajuster `momentum` à 0.8-0.9
4. Installer NumPy

### Oscillations

**Symptôme:** Variance oscille, ne converge jamais

**Solutions:**
1. Augmenter `momentum` (0.8-0.9)
2. Diminuer `strength`
3. Utiliser strength adaptatif (mettre à `None`)

### Gel ne marche pas

**Symptôme:** `freeze_ratio` toujours 0%

**Solutions:**
1. Vérifier `freeze_enabled=True`
2. Relâcher `freeze_threshold` (0.01 → 0.1)
3. Réduire `freeze_stability_steps`

### Erreurs import

**Symptôme:** `ModuleNotFoundError`

**Solution:**
```bash
pip uninstall nexus-cosmic
pip install git+https://github.com/Tryboy869/nexus-cosmic.git
```

---

## Exemples Avancés

### Exemple 1: Consensus Blockchain

```python
from nexus_cosmic import NexusCosmic
import random

class NoeudBlockchain:
    def __init__(self, node_id):
        self.node_id = node_id
        self.valeur_proposee = random.uniform(0, 100)

# Créer 100 nœuds blockchain
noeuds = [NoeudBlockchain(i) for i in range(100)]

# Chaque nœud propose une valeur
system = NexusCosmic(mode='consensus', n_entities=100)

for i, noeud in enumerate(noeuds):
    system.entities[i].state = noeud.valeur_proposee

# Atteindre consensus
result = system.run()

valeur_acceptee = system.get_consensus()

print(f"Tous nœuds s'accordent sur: {valeur_acceptee:.2f}")
print(f"Consensus atteint en {result['steps']} étapes")
```

### Exemple 2: Ordonnanceur Tâches

```python
from nexus_cosmic import NexusCosmic

class Tache:
    def __init__(self, nom, priorite):
        self.nom = nom
        self.priorite = priorite

# Tâches avec priorités
taches = [
    Tache("Bug Critique", 90),
    Tache("Demande Feature", 30),
    Tache("Documentation", 20),
    Tache("Patch Sécurité", 95),
    Tache("Refactoring", 50),
]

# Trier par priorité
priorites = [t.priorite for t in taches]

system = NexusCosmic(mode='sorting', values=priorites)
system.run()

priorites_triees = system.get_sorted_values()

# Réordonner tâches
ordre_taches = [priorites.index(p) for p in priorites_triees]
taches_triees = [taches[i] for i in ordre_taches]

print("Ordre exécution tâches:")
for i, tache in enumerate(taches_triees, 1):
    print(f"{i}. {tache.nom} (priorité {tache.priorite})")
```

### Exemple 3: Fusion Réseau Capteurs

```python
from nexus_cosmic import NexusCosmic
import random

# 50 capteurs température avec bruit
capteurs = [22.0 + random.gauss(0, 2) for _ in range(50)]

# Consensus pour filtrer bruit
system = NexusCosmic(mode='consensus', n_entities=50)

for i, lecture in enumerate(capteurs):
    system.entities[i].state = lecture

result = system.run()

temperature_reelle = system.get_consensus()

print(f"Capteurs individuels: {capteurs[:5]}...")
print(f"Température réelle (consensus): {temperature_reelle:.2f}°C")
print(f"Bruit réduit en {result['steps']} étapes")
```

---

## Contribuer

Voir [CONTRIBUTING.md](CONTRIBUTING.md)

## Licence

Licence MIT - voir [LICENSE](LICENSE)

## Auteur

**Daouda Abdoul Anzize** (Nexus Studio)
- GitHub: [@Tryboy869](https://github.com/Tryboy869)
- Email: nexusstudio100@gmail.com

---

**Bon Calcul avec Nexus-Cosmic !** 🚀
