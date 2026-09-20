from manim import *

class Paqueton(Scene):
    def construct(self):

        # Constantes
        h = 1.0                     # Constante de Planck
        m = 1.0                     # Masa de la partícula
        A = 1                       # amplitud inicial del paquete incidente

        # Parámetros del Paquete
        x0 = -10.0                   # Posición inicial del centro del paquete                               
        E_mean = 7.5                   # Energía promedio <E> del paquetón            
        E_var =  2                 # Varianza de la energía del paquetón

        # Obtenemos nuestra varianza b en k ya conocida, con valor central de l  
        b = m / h**2 * (2*E_mean - np.sqrt(4*E_mean**2 - 2*E_var))     # Varianza de los momentos del paquetón
        l = np.sqrt(E_mean * 2 * m / h**2 - b)                         # Momento central del paquete
        
        tiempo = ValueTracker()                                        # t que se va actualiznaod con la animacion
        t_tot = - 4 * x0 / l 
        
        # Parámetros de la Barrera
        V0 = 8.0                     # alto de la barda
        a  = 2                       # ancho de la barda

        # Calcular Transmitancia de la energía promedio del paquete
        # calculamos el número de onda asociado a la energía media
        k_E_mean = np.sqrt(2*m*E_mean / h**2)

        # - Si E < V0: soluciones evanescentes dentro de la barrera, con κ = sqrt(2m(V0 - E))/h
        # - Si E > V0: soluciones oscilatorias dentro de la barrera, con q = sqrt(2m(E - V0))/h    
        # Utilizamos nuestra formula obtenida para calcular E.
        if E_mean <= V0:
            E_mean = min([E_mean, E_mean - 1e-10])
            # Notita numérica: restamos un epsilon si E == V0 para evitar κ=0 exacto
            kappa_E_mean = np.sqrt(2*m*(V0 - E_mean) / h**2)     
            Transmitancia = (4*k_E_mean**2*kappa_E_mean**2) / (4*k_E_mean**2*kappa_E_mean**2 + (k_E_mean**2 + kappa_E_mean**2)**2 * np.sinh(kappa_E_mean*a)**2)
        elif E_mean > V0:
            q_E_mean = np.sqrt(2*m*(E_mean-V0) / h**2)
            Transmitancia = (4*k_E_mean**2*q_E_mean**2) / (4*k_E_mean**2*q_E_mean**2 + (k_E_mean**2 - q_E_mean**2)**2 * np.sin(q_E_mean*a)**2)

        # Parámetros para la integración numérica (resolución)
        Nk = 10000               # número de puntos "k" para la integración 
        Nx = 800                # número de puntos "x" para el ploteo de Psi(x,t)
        Sigmas = 6.0            # número de desviaciones estándar en k que tomamos alrededor de l

        x_min, x_max = x0*2.5, -x0*2.5
        xs = np.linspace(x_min, x_max, Nx)
        # desviación estandar en k a partir de s (forma gaussiana)
        sigma_k = 1.0 / (2.0 * np.sqrt(b))


        k_min = l - Sigmas * sigma_k            # k minimo de acuerdo al número de desviaciones estandar
        k_max = l + Sigmas * sigma_k            # k maximo de acuerdo al número de desviaciones estandar
        ks = np.linspace(k_min, k_max, Nk)
        dk = ks[1] - ks[0]                      # este sera nuestro paso de integración en k

        #  phi(k), definimos nuestro paquete gaussiano en el espacio de momentos
        def phi_k_vec(ks):
            constante = (2*b/np.pi)**0.25 / np.sqrt(2*b)
            return constante * np.exp(1j * x0*(l-ks)) * np.exp(-(l-ks)**2 / (4.0*b))

        # psi_k(x), calcularemos nuestroa distribución resolviendo el problema de dispersión. 
        # Utilizaremos los coeficientes ya calculados en la anterior entrega
        def psi_k_for_x(ks, x, V0=V0, a=a):
            E_k = (h*ks)**2 / (2.0*m)
            psi_vals = np.zeros_like(ks, dtype=np.complex128)
            
            # Caso E > V0
            mask_prop = E_k > V0
            if np.any(mask_prop):
                k_prop = ks[mask_prop]
                q = np.sqrt(2.0*m*(E_k[mask_prop] - V0)) / h

                # Coeficientes
                den_1 = (k_prop + q)**2 - (q - k_prop)**2 * np.exp(2j * q * a)
                eps = 1e-12
                den_1 = np.where(np.abs(den_1) < eps, eps*np.exp(1j*np.angle(den_1)), den_1)

                B = A * ((k_prop**2 - q**2) * (1 - np.exp(2j*q*a))) / den_1
                C = A * (2.0 * k_prop * (k_prop + q)) / den_1
                D = A * (2.0 * k_prop * (q - k_prop) * np.exp(2j*q*a)) / den_1
                F = A * (4.0 * k_prop * q * np.exp(1j*(q - k_prop) * a)) / den_1

                # Utilizar la psi_k(x) correspondiente (piecewise)
                if x < 0.0:
                    psi_vals[mask_prop] = np.exp(1j * k_prop * x) + B * np.exp(-1j * k_prop * x)
                elif x <= a:
                    psi_vals[mask_prop] = C * np.exp(1j * q * x) + D * np.exp(-1j * q * x)
                else:
                    psi_vals[mask_prop] = F * np.exp(1j * k_prop * x)

            # Caso E <= V0
            mask_ev = ~mask_prop
            if np.any(mask_ev):
                k_ev = ks[mask_ev]
                kappa = np.sqrt(2.0*m*(V0 - E_k[mask_ev])) / h

                den_2 = (k_ev + 1j*kappa)**2 * np.exp(2.0 * kappa * a) - (k_ev - 1j*kappa)**2

                eps = 1e-12
                den_2 = np.where(np.abs(den_2) < eps, eps*np.exp(1j*np.angle(den_2)), den_2)

                B = A * ((k_ev**2 + kappa**2) * (np.exp(2.0 * kappa * a) - 1.0)) / den_2
                C = A * -(2.0 * k_ev * (k_ev - 1j*kappa)) / den_2
                D = A * (2.0 * k_ev * (k_ev + 1j*kappa) * np.exp(2.0 * kappa * a)) / den_2
                F = A * (4j * k_ev * kappa * np.exp(kappa * a) * np.exp(-1j * k_ev * a)) / den_2
                
                # Utilizar la psi_k(x) correspondiente (piecewise)
                if x < 0.0:
                    psi_vals[mask_ev] = np.exp(1j * k_ev * x) + B * np.exp(-1j * k_ev * x)
                elif x <= a:
                    psi_vals[mask_ev] = C * np.exp(kappa * x) + D * np.exp(-kappa * x)
                else:
                    psi_vals[mask_ev] = F * np.exp(1j * k_ev * x)

            return psi_vals
    
        #  Psi(x,t) (integrando con método del rectángulo; lento pero robusto)
        def Psi_of_x_t(t):
            time_phase = np.exp(-1j * h * ks**2 * t / (2.0*m))
            integrand = phi_vals * psi_k_vals * time_phase
            return (1.0/np.sqrt(2.0*np.pi)) * np.sum(integrand, axis=1) * dk
        

        # Calcular Psi(x,t) a lo largo de todo x
        phi_vals = phi_k_vec(ks)
        psi_k_vals = np.array([psi_k_for_x(ks, x) for x in xs])

        # Modulo cuadrado de Psi(x,t)
        def Psi_X_T_2(t):
            Psi_vals = Psi_of_x_t(t)
            prob_density = np.abs(Psi_vals)**2
            return prob_density
        
        # Computar onda inicial y en choque para establecer los ejes
        PSI_0 = Psi_X_T_2(0)
        Y_max = max(PSI_0)

        # Estimación del tiempo en que el centro del paquete llega a la barrera:
        PSI_choque = Psi_X_T_2(-x0 / l)
        Y_choque = max(PSI_choque)

        # Ejes de coordenadas
        eje_psi = Axes(
            x_range = [x_min, x_max, 5],
            y_range=[0, Y_choque*1.15, 0.1],
            x_length=10,
            y_length=4.5,
            tips=False,
        ).shift(0.45*DOWN)

        # Para crear escala del eje de energía
        Altura = 2.5 + eje_psi.c2p(0, Y_choque, 0)[1]
        if 4.5*E_mean / Altura > V0:
            Alto = 4.5*E_mean / Altura
        else:
            Alto = V0

        eje_V = Axes(
            x_range = [x_min, x_max, 5],
            y_range=[0, Alto, 0.1],
            x_length=10,
            y_length=4.5,
            tips=False,
        ).shift(0.45*DOWN)

        # Grafica que se va actualizando
        PSI_2 = always_redraw(lambda: eje_psi.plot_line_graph(
            xs, Psi_X_T_2(tiempo.get_value()), 
            add_vertex_dots=False, line_color = BLUE))

        # Gráfica de la barrera
        V_vals = np.where((0 <= xs) & (xs <= a), V0, 0)
        # Perfil de la barrera
        Barrera = eje_V.plot_line_graph(xs, V_vals, add_vertex_dots=False, line_color = GREEN)
        
        # Rectas numéricas y etiquetas
        X_line = NumberLine(
            x_range=[x_min, x_max, 5],
            font_size=28,
            length=10,
            color=WHITE,
            include_numbers=True,
            label_direction=DOWN,
        ).shift(2.7*DOWN)

        PsiY_line = NumberLine(
            x_range=[0, Y_choque*1.15, 0.1],
            length=4.5,
            font_size=26,
            include_tip=False,
            include_numbers=True,
            rotation=90 * DEGREES,
            color=BLUE,
            label_direction=LEFT
        ).shift(0.45*DOWN + 5.75*LEFT)

        V_line = NumberLine(
            x_range=[0, Alto, 1],
            length=4.5,
            font_size=26,
            include_tip=False,
            include_numbers=True,
            rotation=90 * DEGREES,
            color=GREEN,
            label_direction=RIGHT
        ).shift(0.45*DOWN + 5.75*RIGHT)

        #Titulos y textos de parametros fisicos que usamos

        titulo = MathTex(r"\text{Paquete de Onda Gaussiano en Presencia de Una Barrera de Potencial}", font_size = 38)
        titulo.align_to([0, 3.55, 0], UP)

        parametros = MathTex(rf"\langle k \rangle = {l:.1f} \; | \; \sigma^2_k = {b:.2f} \; | \; \langle E \rangle = {E_mean:.1f} \; | \; \sigma^2_E = {E_var:.2f}  \; | \;", r"V_0", rf" = {V0:.1f} \; | \; a = {a:.1f} \; | \; T ( {E_mean:.1f}) = {Transmitancia:.2f}", font_size = 35)
        parametros.align_to([0, 3, 0], UP)

        X = MathTex(r"\mathbf{x}", font_size = 35).shift(3.55 * DOWN)
        PSI2_text = MathTex(r"|\Psi(x,t)|^2", font_size = 32).shift(5.75 * LEFT + 2.2*UP).set_color(BLUE)
        Potencial = MathTex(r"V(x)", font_size = 32).shift(5.80 * RIGHT + 2.2*UP).set_color(GREEN)

        # Ejes numericos y etiquetas
        self.add(X_line)
        self.add(PsiY_line)
        self.add(V_line)
        self.add(X)
        self.add(PSI2_text, Potencial)

        self.play(Write(VGroup(X_line, X)), Write(VGroup(PsiY_line, PSI2_text)), Write(VGroup(V_line, Potencial)), run_time = 1.5)
        
        # Añadimos superposición de textos físicos + gráfica de psi^2 + barrer
        self.add(titulo, parametros)
        self.add(PSI_2)
        self.add(Barrera)

        # Aseguramos que el eje X quede por encima de la curva
        self.remove(X_line)
        self.add(X_line)
        X_line.set_z_index(999)
        self.play(FadeIn(Group(titulo, parametros, PSI_2, Barrera)))

        # Animación: avanzamos el ValueTracker 'tiempo' de 0 a t_tot
        self.play(tiempo.animate.set_value(t_tot), run_time = t_tot, rate_func = linear)
        self.wait(0.5)