import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import pandas as pd
from scipy.optimize import linprog
import heapq as hp

class Transport_Models:
    class Cell:
        def __init__(self, cost, row=None, col=None):
            self.cost = cost
            self.row = row
            self.col = col

        def __lt__(self, cell):
            return self.cost < cell.cost

        def __eq__(self, cell):
            return self.cost == cell.cost

        def __gt__(self, cell):
            return self.cost > cell.cost

        def __repr__(self):
            return str(self.to_tuple())

        def to_tuple(self):
            return (self.cost, self.row, self.col)

    class Penaltie:
        def __init__(self, val, col):
            self.val = val
            self.col = col

        def __eq__(self, penaltie):
            return self.val == penaltie.val

        def __lt__(self, penaltie):
            return self.val < penaltie.val

        def __gt__(self, penaltie):
            return self.val > penaltie.val

        def to_tuple(self):
            return (self.val, self.col)

        def __repr__(self):
            return str(self.to_tuple())

    @staticmethod
    def get_complete_sol(goal, cof_goal, cof_constrains_ub=None, indpt_ub=None, cof_constrains_eq=None, indpt_eq=None, bounds=None):
        if goal not in ["Max", "Min"]:
            raise Exception("No goal allowed")
        if goal == "Max":
            cof_goal = [-_ for _ in cof_goal]

        bounds = bounds if bounds is not None else [(0, None)] * len(cof_goal)
        case_1 = cof_constrains_ub is not None and indpt_ub is not None
        case_2 = cof_constrains_eq is not None and indpt_eq is not None
        case_3 = case_1 and case_2
        if case_1 and not case_3:
            res = linprog(cof_goal,  # Coeficientes de la funcion objetivo
                          A_ub=cof_constrains_ub,  # Coeficientes de las restricciones
                          b_ub=indpt_ub,  # Terminos independientes de las restricciones
                          bounds=bounds,  # Limites
                          method="highs")  # Metodo recomendado en lugar de simplex
            if res.success:
                val_constrains = [sum(a*x for a, x in zip(constrain, res.x)) for constrain in cof_constrains_ub]
                clrncs = [abs(indpt_ub[i]-val_constrains[i]) for i in range(len(indpt_ub))]
                active_val_constrains_type = [i == 0 for i in clrncs]
                sols = {
                    "vars_sols": np.array(list(res.x)),
                    "fun_obj": res.fun,
                    "slacks": clrncs,
                    "sens": Transport_Models.get_sens_anyls(cof_goal, cof_constrains_ub, indpt_ub),
                    "active_constrains": active_val_constrains_type
                }
                return sols
            else:
                return f"No optimal solution found: {res.message}"

        if case_2 and not case_3:
            res = linprog(cof_goal,  # Coeficientes de la funcion objetivo
                          A_eq=cof_constrains_eq,  # Coeficientes de las restricciones
                          b_eq=indpt_eq,  # Terminos independientes de las restricciones
                          bounds=bounds,  # Limites
                          method="highs")  # Metodo recomendado en lugar de simplex
            if res.success:
                val_constrains = [sum(a*x for a, x in zip(constrain, res.x)) for constrain in cof_constrains_eq]
                clrncs = [abs(indpt_eq[i]-val_constrains[i]) for i in range(len(indpt_eq))]
                active_val_constrains_type = [i == 0 for i in clrncs]
                sols = {
                    "vars_sols": np.array(list(res.x)),
                    "fun_obj": res.fun,
                    "slacks": clrncs,
                    "sens": Transport_Models.get_sens_anyls(cof_goal, cof_constrains_eq, indpt_eq),
                    "active_constrains": active_val_constrains_type
                }
                return sols
            else:
                return f"No optimal solution found: {res.message}"

        if case_3:
            res = linprog(cof_goal,  # Coeficientes de la funcion objetivo
                          A_ub=cof_constrains_ub,
                          b_ub=indpt_ub,
                          A_eq=cof_constrains_eq,  # Coeficientes de las restricciones
                          b_eq=indpt_eq,  # Terminos independientes de las restricciones
                          bounds=bounds,  # Limites
                          method="highs")  # Metodo recomendado en lugar de simplex

            if res.success:
                cof_constrains = cof_constrains_ub + cof_constrains_eq
                indpt = indpt_eq + indpt_ub
                val_constrains = [sum(a*x for a, x in zip(constrain, res.x)) for constrain in cof_constrains]
                clrncs = [abs(indpt[i]-val_constrains[i]) for i in range(len(indpt))]
                active_val_constrains_type = [i == 0 for i in clrncs]
                sols = {
                    "vars_sols": np.array(list(res.x)),
                    "fun_obj": res.fun,
                    "slacks": clrncs,
                    "sens": Transport_Models.get_sens_anyls(cof_goal, cof_constrains, indpt),
                    "active_constrains": active_val_constrains_type
                }
                return sols
            else:
                return f"No optimal solution found: {res.message}"

    @staticmethod
    def get_sens_anyls(cof_goal, cof_constrains, indpt):
        cof_goal_dual = np.array(indpt)  # Coeficientes duales son los límites originales
        cof_constrains_dual = np.array(cof_constrains).T  # Transposición para dual
        cof_constrains_dual_neg = -cof_constrains_dual  # Negar restricciones para "menor o igual"
        indpt_dual = -np.array(cof_goal)  # Límite independiente dual
        res_dual = linprog(cof_goal_dual,
                           A_ub=cof_constrains_dual_neg,
                           b_ub=indpt_dual,
                           method="highs")
        if not res_dual.success:
            raise Exception(f"No optimal solution for dual: {res_dual.message}")
        val_constrains = [sum(a*x for a, x in zip(constrain, res_dual.x)) for constrain in cof_constrains_dual]
        clrncs = [abs(indpt[i]-val_constrains[i]) for i in range(len(indpt))]
        active_val_constrains_type = [i == 0 for i in clrncs]
        return {
            "shade_price": res_dual.x,
            "fun_obj": res_dual.fun,
            "slacks": clrncs,
            "active_constrains": active_val_constrains_type
        }

    def __init__(self, costs, supply, demand):
        self.costs = np.array(costs.copy())
        self.supply = supply.copy()
        self.demand = demand.copy()

    def balance_model(self):
        total_supply = sum(self.supply)
        total_demand = sum(self.demand)
        supply = self.supply.copy()
        demand = self.demand.copy()
        costs = self.costs.copy()

        if total_supply > total_demand:
            # Agregar una columna ficticia a la demanda
            demand.append(total_supply - total_demand)
            # Agregar una columna de ceros a cada fila en costs
            costs = np.column_stack((costs, np.zeros(len(costs))))
        elif total_demand > total_supply:
            # Agregar una fila ficticia a la oferta
            supply.append(total_demand - total_supply)
            # Agregar una fila de ceros a costs
            costs = np.vstack((costs, np.zeros(len(demand))))

        self.costs = costs
        self.supply = supply
        self.demand = demand
        return self.costs, self.supply, self.demand

    def total_costs(self, sol_asign):
        total = 0
        for i, row_sol in enumerate(sol_asign):
            total += sum([assign * cost for assign, cost in zip(row_sol, self.costs[i])])
        return total

    def general_method(self, bounds=None):
        costs, supply, demand = self.balance_model()
        n_rows, n_cols = costs.shape
        bounds = bounds if bounds is not None else [(0, None)] * (n_rows * n_cols)
        cof_goal = costs.flatten()

        supply_matrix_cof = np.zeros((n_rows, n_rows * n_cols))
        for i in range(n_rows):
            supply_matrix_cof[i, i * n_cols: (i + 1) * n_cols] = 1

        demand_matrix_cof = np.zeros((n_cols, n_rows * n_cols))
        for j in range(n_cols):
            demand_matrix_cof[j, j::n_cols] = 1

        cof_constrains = np.vstack([supply_matrix_cof, demand_matrix_cof])  # apilar
        indpt = np.concatenate([supply, demand])  # unir
        res = Transport_Models.get_complete_sol("Min",
                                                cof_goal,
                                                cof_constrains_eq=cof_constrains,
                                                indpt_eq=indpt)
        if isinstance(res, str):
            raise Exception(res)
        sol = res["vars_sols"].reshape(n_rows, n_cols)
        return {"allocation_matrix": sol, "total_cost": self.total_costs(sol), "method": "simplex_method"}

    def northwest_corner(self):
        # Balancing the model
        costs, supply, demand = self.balance_model()
        rows, cols = len(supply), len(demand)
        sol = np.zeros((rows, cols))
        i, j = 0, 0

        while i < rows and j < cols:
            asignated_val = min(supply[i], demand[j])
            sol[i][j] = asignated_val
            supply[i] -= asignated_val
            demand[j] -= asignated_val

            if supply[i] == 0:
                i += 1
            elif demand[j] == 0:
                j += 1

        return {"allocation_matrix": sol, "total_cost": self.total_costs(sol), "method": "northwest_corner"}

    def min_cost(self):
        costs, supply, demand = self.balance_model()
        rows, cols = len(supply), len(demand)
        sol = np.zeros((rows, cols))

        # Obtener objetos y ordenar
        cells = [self.Cell(costs[i][j], i, j) for i in range(rows) for j in range(cols)]
        cells.sort()  # Ordenar por costo (menor a mayor)
        for cell in cells:
            cost, i, j = cell.to_tuple()
            if supply[i] > 0 and demand[j] > 0:
                asignated_val = min(supply[i], demand[j])
                sol[i][j] = asignated_val
                supply[i] -= asignated_val
                demand[j] -= asignated_val

        return {"allocation_matrix": sol, "total_cost": self.total_costs(sol), "method": "min_cost"}

    def vogel(self):
        costs, supply, demand = self.balance_model()
        rows, cols = len(supply), len(demand)
        sol = np.zeros((rows, cols))

        available_cols = list(range(cols))
        available_rows = list(range(rows))

        while available_cols and available_rows:
            penalties = []
            for col in available_cols:
                cost_list = [costs[row][col] for row in available_rows]
                if len(cost_list) >= 2:
                    sorted_costs = sorted(cost_list)
                    penalty = sorted_costs[1] - sorted_costs[0]
                elif len(cost_list) == 1:
                    penalty = cost_list[0]
                else:
                    penalty = 0
                penalties.append((penalty, col))

            # Seleccionar la columna con mayor penalidad
            penalties.sort(reverse=True)
            selected_col = penalties[0][1]

            # Encontrar la fila con el costo mínimo en la columna seleccionada
            min_cost = float('inf')
            selected_row = None
            for row in available_rows:
                if costs[row][selected_col] < min_cost:
                    min_cost = costs[row][selected_col]
                    selected_row = row

            # Asignar el mínimo entre oferta y demanda
            asignated_val = min(supply[selected_row], demand[selected_col])
            sol[selected_row][selected_col] = asignated_val
            supply[selected_row] -= asignated_val
            demand[selected_col] -= asignated_val

            # Eliminar fila o columna si se cumple la oferta o demanda
            if supply[selected_row] == 0 and demand[selected_col] == 0:
                available_rows.remove(selected_row)
                available_cols.remove(selected_col)
            elif supply[selected_row] == 0:
                available_rows.remove(selected_row)
            elif demand[selected_col] == 0:
                available_cols.remove(selected_col)

        return {"allocation_matrix": sol, "total_cost": self.total_costs(sol), "method": "vogel"}

# Función para guardar matrices en archivos CSV
def save_to_csv(costs, supply, demand):
    try:
        # Guardar la matriz de costos
        costs_df = pd.DataFrame(costs)
        costs_file = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], title="Guardar Matriz de Costos")
        if costs_file:
            costs_df.to_csv(costs_file, index=False, header=False)
            messagebox.showinfo("Éxito", "Matriz de costos guardada correctamente.")

        # Guardar la oferta
        supply_df = pd.DataFrame({"Oferta": supply})
        supply_file = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], title="Guardar Oferta")
        if supply_file:
            supply_df.to_csv(supply_file, index=False)
            messagebox.showinfo("Éxito", "Oferta guardada correctamente.")

        # Guardar la demanda
        demand_df = pd.DataFrame({"Demanda": demand})
        demand_file = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], title="Guardar Demanda")
        if demand_file:
            demand_df.to_csv(demand_file, index=False)
            messagebox.showinfo("Éxito", "Demanda guardada correctamente.")

    except Exception as e:
        messagebox.showerror("Error", f"Se produjo un error al guardar los archivos: {e}")

# Función para manejar el cálculo y mostrar los resultados
def solve_transport_problem():
    try:
        # Leer datos de la interfaz
        costs = cost_entry.get("1.0", "end-1c").strip()
        supply = supply_entry.get("1.0", "end-1c").strip()
        demand = demand_entry.get("1.0", "end-1c").strip()

        # Validar que todos los campos estén completos
        if not costs or not supply or not demand:
            messagebox.showwarning("Datos incompletos", "Debe rellenar los datos.")
            return

        # Convertir las entradas a listas y matrices
        try:
            costs = [[float(num) for num in line.split()] for line in costs.split("\n") if line.strip()]
            supply = [float(num) for num in supply.split() if num.strip()]
            demand = [float(num) for num in demand.split() if num.strip()]
        except ValueError:
            messagebox.showerror("Error de Formato", "Asegúrese de que todos los valores sean números separados por espacios.")
            return

        # Crear instancia del modelo
        model = Transport_Models(costs, supply, demand)

        # Seleccionar método
        method = method_var.get()
        if method == "Esquina Noroeste":
            result = model.northwest_corner()
        elif method == "Costo Mínimo":
            result = model.min_cost()
        elif method == "Vogel":
            result = model.vogel()
        elif method == "Método General":
            result = model.general_method()
        else:
            raise ValueError("Método no válido.")

        # Mostrar resultados
        allocation = result["allocation_matrix"]
        total_cost = result["total_cost"]
        output_text.set(f"Método: {result['method']}\n"
                        f"Matriz de Asignación:\n{allocation}\n"
                        f"Costo Total: {total_cost}")

        # Preguntar si desea guardar los datos
        if messagebox.askyesno("Guardar datos", "¿Desea guardar los datos en archivos CSV?"):
            save_to_csv(model.costs, model.supply, model.demand)

    except Exception as e:
        messagebox.showerror("Error", f"Se produjo un error: {e}")

# Crear la interfaz gráfica
root = tk.Tk()
root.title("Problema de Transporte")
root.geometry("500x600")  # Tamaño de la ventana

# Crear los widgets de la interfaz

# Matriz de Costos
cost_label = tk.Label(root, text="Matriz de Costos (separada por espacios):")
cost_label.pack(pady=(10, 0))

cost_entry = tk.Text(root, height=10, width=50)
cost_entry.pack(pady=(0, 10))

# Oferta
supply_label = tk.Label(root, text="Oferta (separada por espacios):")
supply_label.pack(pady=(10, 0))

supply_entry = tk.Text(root, height=2, width=50)
supply_entry.pack(pady=(0, 10))

# Demanda
demand_label = tk.Label(root, text="Demanda (separada por espacios):")
demand_label.pack(pady=(10, 0))

demand_entry = tk.Text(root, height=2, width=50)
demand_entry.pack(pady=(0, 10))

# Selección del Método
method_label = tk.Label(root, text="Selecciona el Método:")
method_label.pack(pady=(10, 0))

method_var = tk.StringVar()
methods_combobox = ttk.Combobox(root, textvariable=method_var, state="readonly")
methods_combobox["values"] = ["Esquina Noroeste", "Costo Mínimo", "Vogel", "Método General"]
methods_combobox.current(0)  # Seleccionar por defecto "Esquina Noroeste"
methods_combobox.pack(pady=(0, 10))

# Botón para Resolver
solve_button = tk.Button(root, text="Resolver", command=solve_transport_problem, bg="blue", fg="white", font=("Helvetica", 12, "bold"))
solve_button.pack(pady=(10, 20))

# Área para Mostrar Resultados
output_label = tk.Label(root, text="Resultados:", font=("Helvetica", 12, "bold"))
output_label.pack()

output_text = tk.StringVar()
output_display = tk.Label(root, textvariable=output_text, justify="left", bg="white", relief="sunken", anchor="nw", width=60, height=15)
output_display.pack(padx=10, pady=(0, 10), fill="both", expand=True)

# Ejecutar la interfaz gráfica
root.mainloop()
