### Ordnerstruktur

actor_critic.py : Actor Critic Implementierung
DDPG.py : DDPG Implementierung
TD3.py : TD3 Implementierung

agent.py : Helper Classes, Methods

tune_*.py : Hyperparmeter Tuning der jeweligen Methoden

Eval Results/  : Evaluation Ergebnisse der jeweligen Methoden
Hyperparameter Tuning/ Hyperparamter Tuning logs der jeweligen Methoden





Bibliotheken Installieren :

```
pip install -r requirements.txt
```


**Um alle Methoden (Actor Critic, TD3, DDPG) auszuführen, führen Sie „main.py“ aus** 

```
python main.py
```

Um einzelnen Methoden zum laufen :

- Actor - Critic
 -> `python actor_critic.py`

- DDPG
 -> `python DDPG.py`

- TD3
 -> `python TD3.py`