@doc doc"""
Minimise une fonction en utilisant l'algorithme des régions de confiance avec
    - le pas de Cauchy
ou
    - le pas issu de l'algorithme du gradient conjugue tronqué

# Syntaxe
```julia
xk, nb_iters, f(xk), flag = Regions_De_Confiance(algo,f,gradf,hessf,x0,option)
```

# Entrées :

   * **algo**        : (String) string indicant la méthode à utiliser pour calculer le pas
        - **"gct"**   : pour l'algorithme du gradient conjugué tronqué
        - **"cauchy"**: pour le pas de Cauchy
   * **f**           : (Function) la fonction à minimiser
   * **gradf**       : (Function) le gradient de la fonction f
   * **hessf**       : (Function) la hessiene de la fonction à minimiser
   * **x0**          : (Array{Float,1}) point de départ
   * **options**     : (Array{Float,1})
     * **deltaMax**      : utile pour les m-à-j de la région de confiance
                      ``R_{k}=\left\{x_{k}+s ;\|s\| \leq \Delta_{k}\right\}``
     * **gamma1,gamma2** : ``0 < \gamma_{1} < 1 < \gamma_{2}`` pour les m-à-j de ``R_{k}``
     * **eta1,eta2**     : ``0 < \eta_{1} < \eta_{2} < 1`` pour les m-à-j de ``R_{k}``
     * **delta0**        : le rayon de départ de la région de confiance
     * **max_iter**      : le nombre maximale d'iterations
     * **Tol_abs**       : la tolérence absolue
     * **Tol_rel**       : la tolérence relative

# Sorties:

   * **xmin**    : (Array{Float,1}) une approximation de la solution du problème : ``min_{x \in \mathbb{R}^{n}} f(x)``
   * **fxmin**   : (Float) ``f(x_{min})``
   * **flag**    : (Integer) un entier indiquant le critère sur lequel le programme à arrêter
      - **0**    : Convergence
      - **1**    : stagnation du ``x``
      - **2**    : stagnation du ``f``
      - **3**    : nombre maximal d'itération dépassé
   * **nb_iters** : (Integer)le nombre d'iteration qu'à fait le programme

# Exemple d'appel
```julia
algo="gct"
f(x)=100*(x[2]-x[1]^2)^2+(1-x[1])^2
gradf(x)=[-400*x[1]*(x[2]-x[1]^2)-2*(1-x[1]) ; 200*(x[2]-x[1]^2)]
hessf(x)=[-400*(x[2]-3*x[1]^2)+2  -400*x[1];-400*x[1]  200]
x0 = [1; 0]
options = []
xmin, fxmin, flag,nb_iters = Regions_De_Confiance(algo,f,gradf,hessf,x0,options)
```
"""
function Regions_De_Confiance(algo,f::Function,gradf::Function,hessf::Function,x0,options, nb_iter_TGC = 100)
    #choix du pas 
    pas = Gradient_Conjugue_Tronque
    if algo == "cauchy"  
        pas = Pas_De_Cauchy
    elseif algo == "gctb"  
        pas = Gradient_Conjugue_Tronque_bis 
    end
    if options == []
        deltaMax = 10
        gamma1 = 0.5
        gamma2 = 2.00
        eta1 = 0.25
        eta2 = 0.75
        delta0 = 2
        max_iter = 20000
        Tol_abs = sqrt(eps())
        Tol_rel = [1e-12,1e-15]
    else
        deltaMax = options[1]
        gamma1 = options[2]
        gamma2 = options[3]
        eta1 = options[4]
        eta2 = options[5]
        delta0 = options[6]
        max_iter = options[7]
        Tol_abs = options[8]
        Tol_rel = options[9]
    end

    n = length(x0)
    xmin = zeros(n)
    fxmin = f(xmin)
    flag = 0
    nb_iters = 0
    #initialisation et calcul du premier pas 
    x = copy(x0)
    f0 = f(x)
    grad0 = gradf(x)
    hess0 = hessf(x)
    delta = delta0
    s=0
    if algo == "cauchy"
        s = pas(grad0, hess0, delta)[1]
    else
        s = pas(grad0, hess0, [delta, nb_iter_TGC, 1e-6])[1] 
    end
    #calcul de rho initial
    rho = (f0 - f(x + s))/(transpose(grad0)*s + 1/2 * transpose(s) * hess0 * s)
    #mise à jour de l'itéré courant en fonction des paramètres eta pour la premiere iteration
    if rho >= eta1 
        x_next = x + s
        f_next = f(x_next)
        conditions = [false, norm(x_next - x) <= Tol_rel[1]*(norm(x) + Tol_abs), norm(f_next - f0)<= Tol_rel[2]*(norm(f0)+Tol_abs), nb_iters == max_iter] 
    else 
        x_next = x
        f_next = f(x_next)
        conditions = [false, nb_iters == max_iter] 
    end 
    
    #mise à jour de la région de confiance pour la premiere iteration 
    if rho >= eta2 
        delta = min(gamma2*delta, deltaMax)
    elseif rho <= eta1 
        delta = gamma1 * delta
    end  

    #verification des conditions pour la premiere iteration
    condition_verif = false
    for cond in conditions 
        if cond == true 
            condition_verif = true
        end
    end
    while !condition_verif 
        x = copy(x_next)

        f_prec = f(x)
        grad = gradf(x)
        hess = hessf(x)
        if algo == "cauchy"
            s = pas(grad, hess, delta)[1]
        else 
            s = pas(grad, hess, [delta, nb_iter_TGC, 1e-6])[1]
        end
        #calcul de rho
        rho = -(f(x) - f(x + s))/(transpose(grad)*s + 1/2 * transpose(s)*hess*s)

        #mise à jour de l'itéré courant en fonction des paramètres eta
        if rho >= eta1 
            x_next = x + s
            f_next = f(x_next)
            conditions = [norm(grad) <= Tol_rel[1]*(norm(grad0)+Tol_abs), norm(x_next - x) <= Tol_rel[1]*(norm(x) + Tol_abs), norm(f_next - f_prec)<= Tol_rel[2]*(norm(f_prec)+Tol_abs), nb_iters == max_iter] 
        else 
            x_next = x
            f_next = f(x_next)
            conditions = [norm(grad) <= Tol_rel[1]*(norm(grad0)+Tol_abs), nb_iters == max_iter] 
        end 
        #mise à jour de la région de confiance 
        if rho >= eta2 
            delta = min(gamma2*delta, deltaMax)
        elseif rho <= eta1 
            delta = gamma1 * delta
        end 

        #verification des conditions pour la premiere iteration
        for cond in conditions 
            if cond == true 
                condition_verif = true
            end
        end
        nb_iters += 1
    end
    #flag de sortie de boucle 
    for i in 1:length(conditions) 
        if conditions[i] == true 
            flag = i-1
        end
    end
    xmin = x_next
    fxmin = f_next
    return xmin, fxmin, flag, nb_iters
end
