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

function Gradient_Conjugue_Tronque_bis(gradfk,hessfk,options)
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
    j = 0
    fin = false # boolen permettant de mettre fin a a la boucle 
    
    flag_tgc = 0 # variable qui premet de voir quel conditiond d'arret on observe 
    CN = tol * ng # condition d'arret de convergence 
    # si gradient nul on rentre pas dans la boucle
    if ng == 0 
        fin = true 
    end 
    while j < max_iter && !fin
        j += 1 
        kj = transpose(pj) * hessfk * pj
        alphaj = (transpose(gj) * gj) / kj
        #première condition d'arrêt
        if kj <= 0
            #on regarde si c'est la première itération 
            if j == 1
                sig_pterme = -transpose(sj)*pj / norm(pj)^2
                sig_dterme = sqrt((transpose(sj)*pj)^2 - norm(pj)^2*(norm(sj)^2 - deltak^2))/norm(pj)^2
                sigma1 = sig_pterme - sig_dterme
                sigma2 = sig_pterme + sig_dterme
                # calcul de la plus petite des solutions et on incremente le pas
                if q(sj + sigma1 * pj, gj, hessfk) < q(sj + sigma2 * pj, gj, hessfk)
                    sj = sj + sigma1 * pj 
                else 
                    sj = sj + sigma2 * pj
                end
            
            end
            # on sort de la boucle en retenant le dernier pas, sauf si c'est la 1ere itération, et on indique la condition d'arret(1) .
            flag_tgc = 1
            fin = true
            s= sj
        
        # Deuxieme condition
        elseif norm(sj + alphaj * pj) >= deltak
            #on regarde si c'est la première itération
            if j ==1
                sig_pterme = -transpose(sj)*pj / norm(pj)^2
                sig_dterme = sqrt((transpose(sj)*pj)^2 - norm(pj)^2*(norm(sj)^2 - deltak^2))/norm(pj)^2
                sigma = sig_pterme + sig_dterme
                sj= sj + sigma * pj
                #on calcule la solution et on incremente le pas
            end
            # on sort de la boucle en retenant le dernier pas, sauf si c'est la 1ere itération, et on indique la condition d'arret(2) .
            flag_tgc = 2
            fin = true
            s = sj 
        else
            # si aucune condition d'arret n'est retenu on incremente toute les variables
            sj = sj +alphaj * pj
            gjp = copy(gj +alphaj *hessfk *pj)
            Bj = transpose(gjp) * gjp/ (transpose(gj) * gj)
            pjp = copy(-gjp + Bj * pj)
            gj = copy(gjp)
            pj = copy(pjp)
        
        # condition de convergence : on sort de la boucle si l'algo converge et on indique la condition d'arret (3)
            if norm(gj) <= CN 
                fin = true 
                flag_tgc = 3
                s = sj
            end
        end

    end
    # on retourne le dernier pas choisis et le flag d'arret 
    return s, flag_tgc
end 