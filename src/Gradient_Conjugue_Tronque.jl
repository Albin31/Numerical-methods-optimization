@doc doc"""
Minimise le problème : ``min_{||s||< \delta_{k}} q_k(s) = s^{t}g + (1/2)s^{t}Hs``
                        pour la ``k^{ème}`` itération de l'algorithme des régions de confiance

# Syntaxe
sk = Gradient_Conjugue_Tronque(fk,gradfk,hessfk,option)

# Entrées :   
   * *gradfk*           : (Array{Float,1}) le gradient de la fonction f appliqué au point xk
   * *hessfk*           : (Array{Float,2}) la Hessienne de la fonction f appliqué au point xk
   * *options*          : (Array{Float,1})
      - *delta*    : le rayon de la région de confiance
      - *max_iter* : le nombre maximal d'iterations
      - *tol*      : la tolérance pour la condition d'arrêt sur le gradient


# Sorties:
   * *s* : (Array{Float,1}) le pas s qui approche la solution du problème : ``min_{||s||< \delta_{k}} q(s)``

# Exemple d'appel:
gradf(x)=[-400*x[1]*(x[2]-x[1]^2)-2*(1-x[1]) ; 200*(x[2]-x[1]^2)]
hessf(x)=[-400*(x[2]-3*x[1]^2)+2  -400*x[1];-400*x[1]  200]
xk = [1; 0]
options = []
s = Gradient_Conjugue_Tronque(gradf(xk),hessf(xk),options)
"""

function q(s, g, H) 
    return transpose(g) * s + 1/2*transpose(s) * H * s
end 

function Gradient_Conjugue_Tronque(gradfk,hessfk,options)
    "# Si option est vide on initialise les 3 paramètres par défaut"
    if options == []
        deltak = 2
        max_iter = 100
        tol = 1e-6
    else
        deltak = options[1]
        max_iter = options[2]
        tol = options[3]
    end

    n = length(gradfk)
    s = zeros(n)
    sj = copy(s)

    g_ini = gradfk
    gj = copy(g_ini)
    ng = norm(g_ini)

    p_ini = -gradfk
    pj = copy(p_ini)
    j = 1
    
    fin = false  # booleen permettant de mettre fin a a la boucle 

    CN = tol * ng # condition d'arret de convergence 
    flag_tgc = 0 #condition d'arret elle nous indiquera pourquoi nous sortons de la boucle 
    # si gradient nul on rentre pas dans la boucle
    if ng == 0
        fin = true 
    end 
    while j <= max_iter && !fin
        kj = transpose(pj) * hessfk * pj
        alphaj = (transpose(gj) * gj) / kj

        #cas concave
        if kj <= 0
            # on a 2 pas possible, ils sont sur le bord de la région de confiance
            sig_pterme = -transpose(sj)*pj / norm(pj)^2
            sig_dterme = sqrt((transpose(sj)*pj)^2 - norm(pj)^2*(norm(sj)^2 - deltak^2))/norm(pj)^2
            sigma1 = sig_pterme - sig_dterme # pas 1
            sigma2 = sig_pterme + sig_dterme # pas 2
            fin = true # on met fin à la boucle
            flag_tgc = 1 # on indique la condition d'arret 

            # on choisit le pas qui minimise le plus notre fonction et on le stock dans s
            if q(sj + sigma1 * pj, gj, hessfk) < q(sj + sigma2 * pj, gj, hessfk)
                s = sj + sigma1 * pj
            else 
                s = sj + sigma2 * pj
            end
        # min global hors de la région de confiance 
        elseif norm(sj + alphaj * pj) >= deltak 
            # on calcule le pas et on le stocke dans s
            sig_pterme = -transpose(sj)*pj / norm(pj)^2
            sig_dterme = sqrt((transpose(sj)*pj)^2 - norm(pj)^2*(norm(sj)^2 - deltak^2))/norm(pj)^2
            sigma = sig_pterme + sig_dterme
            s = sj + sigma * pj
            fin = true  # on met fin à la boucle
            flag_tgc = 2# on indique la condition d'arret 
        else 
            # le min global est toujours dans la région de confiance
            # on calcule le nouveau pas et on in toutes les variables et 
            # les directions de recherche du min
            sj = sj +alphaj * pj
            gjp = copy(gj +alphaj *hessfk *pj)
            Bj = transpose(gjp) * gjp/ (transpose(gj) * gj)
            pjp = copy(-gjp + Bj * pj)
            gj = copy(gjp)
            pj = copy(pjp)
            j += 1
            s = sj
        end
        #condition de convergence :  # condition de convergence : on sort de la boucle si l'algo converge et on indique la condition d'arret (3)

        if norm(gj) <= CN 
            fin = true 
            s = sj
            flag_tgc = 3 # on indique la condition d'arret 
        end 
    end
    # on retourne le dernier pas choisis et le flag d'arret
    return s, flag_tgc 
end 