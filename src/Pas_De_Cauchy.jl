@doc doc"""
Approximation de la solution du sous-problème ``q_k(s) = s^{t}g + (1/2)s^{t}Hs`` 
        avec ``s=-t g_k,t > 0,||s||< \delta_k ``


# Syntaxe
```julia
s1, e1 = Pas_De_Cauchy(gradient,Hessienne,delta)
```

# Entrées
 * **gradfk** : (Array{Float,1}) le gradient de la fonction f appliqué au point ``x_k``
 * **hessfk** : (Array{Float,2}) la Hessienne de la fonction f appliqué au point ``x_k``
 * **delta**  : (Float) le rayon de la région de confiance

# Sorties
 * **s** : (Array{Float,1}) une approximation de la  solution du sous-problème
 * **e** : (Integer) indice indiquant l'état de sortie:
        si g != 0
            si on ne sature pas la boule
              e <- 1
            sinon
              e <- -1
        sinon
            e <- 0

# Exemple d'appel
```julia
g1 = [0; 0]
H1 = [7 0 ; 0 2]
delta1 = 1
s1, e1 = Pas_De_Cauchy(g1,H1,delta1)
```
"""
function Pas_De_Cauchy(g,H,delta)

  e = 0  #Flag
  n = length(g)
  s = zeros(n)
  coeff = transpose(g) * H * g

  norm_g = norm(g,2)
  #norme nulle on renvoit s ( = 0)
  if norm(g) == 0
      t = -1
      e = -1
  # cas convexe 
  elseif coeff > 0

    # minimum entre les 2 bornes 
      aux = (norm_g ^2) / coeff
      if aux < delta / norm_g
        # on ne sature pas la boule 
        t = aux
        e = 1
      else
        # boule saturé
        t = delta / norm_g
        e =  -1
      end
  else
     # cas concave, boule saturé
      t= delta / norm_g
      e= -1
  end
  
  # on calcul le pas selon notre position et on le renvoit
  s = -t * g
  return s, e
end
