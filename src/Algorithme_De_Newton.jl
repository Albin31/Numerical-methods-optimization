@doc doc"""
Approximation de la solution du problème ``\min_{x \in \mathbb{R}^{n}} f(x)`` en utilisant l'algorithme de Newton

# Syntaxe
```julia
xk,f_min,flag,nb_iters = Algorithme_de_Newton(f,gradf,hessf,x0,option)
```

# Entrées :
   * **f**       : (Function) la fonction à minimiser
   * **gradf**   : (Function) le gradient de la fonction f
   * **hessf**   : (Function) la Hessienne de la fonction f
   * **x0**      : (Array{Float,1}) première approximation de la solution cherchée
   * **options** : (Array{Float,1})
       * **max_iter**      : le nombre maximal d'iterations
       * **Tol_abs**       : la tolérence absolue
       * **Tol_rel**       : la tolérence relative

# Sorties:
   * **xmin**    : (Array{Float,1}) une approximation de la solution du problème  : ``\min_{x \in \mathbb{R}^{n}} f(x)``
   * **f_min**   : (Float) ``f(x_{min})``
   * **flag**     : (Integer) indique le critère sur lequel le programme à arrêter
      * **0**    : Convergence
      * **1**    : stagnation du xk
      * **2**    : stagnation du f
      * **3**    : nombre maximal d'itération dépassé
   * **nb_iters** : (Integer) le nombre d'itérations faites par le programme

# Exemple d'appel
```@example
using Optinum
f(x)=100*(x[2]-x[1]^2)^2+(1-x[1])^2
gradf(x)=[-400*x[1]*(x[2]-x[1]^2)-2*(1-x[1]) ; 200*(x[2]-x[1]^2)]
hessf(x)=[-400*(x[2]-3*x[1]^2)+2  -400*x[1];-400*x[1]  200]
x0 = [1; 0]
options = []
xmin,f_min,flag,nb_iters = Algorithme_De_Newton(f,gradf,hessf,x0,options)
```
"""
function Algorithme_De_Newton(f::Function,gradf::Function,hessf::Function,x0,options)
    "# Si options == [] on prends les paramètres par défaut"
    if options == []
        max_iter = 100
        Tol_abs = sqrt(eps())
        Tol_rel = [1e-15, 1e-15]
    else
        max_iter = options[1]
        Tol_abs = options[2]
        Tol_rel = options[3]
    end

    n = length(x0)
    xmin = zeros(n)
    f_min = 0
    flag = 0 # flag pour connaitre la condition de sortie de boucle
    nb_iters = 0

    #première initialisation des variables dont on aura besoin
    x = copy(x0)
    f0 = f(x)
    grad0 = gradf(x)
    hess0 = hessf(x)
    dk = hess0 \ (-grad0)
    x_next = x + dk
    f_next = f(x_next)

    conditions = [false, norm(x_next - x) <= Tol_rel[1]*(norm(x) + Tol_abs), norm(f_next - f0)<= Tol_rel[2]*(norm(f0)+Tol_abs), nb_iters == max_iter] 
    #condition d'arret de la boucle, toutes contenues dans une liste
    condition_verif = false # critère de fin si une des conditions devient vraie alors il deviendra vrai et la boucle prendra fin 
    for cond in conditions 
        if cond == true 
            condition_verif = true
        end
    end

    while !condition_verif 
        x = copy(x_next) #on met x a jour
        # on met le gradien la hessienne et la fonction a jour
        f_prec = f(x)
        grad = gradf(x)
        hess = hessf(x)
        dk = hess \ (-grad)
        x_next = x + dk # calcul du prochain x
        f_next = f(x_next) # calcul du prochain f
        nb_iters += 1
        #on incremente les conditions
        conditions = [norm(grad) <= Tol_rel[1]*(norm(grad0)+Tol_abs), norm(x_next - x) <= Tol_rel[2]*(norm(x) + Tol_abs), norm(f_next - f_prec)<= Tol_rel[2]*(norm(f_prec)+Tol_abs), nb_iters == max_iter]
        for cond in conditions # on teste toutes nos conditions d'arret
            if cond == true 
                condition_verif = true # si une condition est vérifiée alors on sort de la boucle
            end
        end
    end 
    for i in 1:length(conditions)
        # on met le flag a jour si une condition d'arret est verifiée 
        if conditions[i] == true 
            flag = i-1
        end
    end
    # on stocke les meilleures valeurs trouvées pour le min 
    xmin = x_next
    f_min = f_next
    # on renvoit les meilleures valeurs trouvées la condition d'arret et le nombre d'iteration
    return xmin,f_min,flag,nb_iters
end
