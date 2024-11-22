import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import pandas as pd
from scipy.optimize import linprog
import heapq as hp

class Transport_Models:
  class Cell:
    def __init__(self, cost, row = None, col = None):
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
  def get_complete_sol(goal, cof_goal, cof_constrains_ub = None, indpt_ub = None, cof_constrains_eq = None, indpt_eq = None, bounds = None):
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
                    A_ub= cof_constrains_ub, # Coeficientes de las restricciones
                    b_ub= indpt_ub, # Terminos independientes de las restricciones
                    bounds = bounds, # Limites
                    method = "highs") # Metodo recomendado en lugar de simplex
      if res.success:
        val_constrains = [sum(a*x for a, x in zip(constrain, res.x)) for constrain in cof_constrains_ub]
        clrncs = [abs(indpt_ub[i]-val_constrains[i]) for i in range(0,len(indpt_ub))]
        active_val_constrains_type = [i == 0 for i in clrncs]
        sols = {"vars_sols"         : np.array(list(res.x)),
                "fun_obj"           : res.fun,
                "slacks"            : clrncs,
                "sens"              : Transport_Models.get_sens_anyls(cof_goal, cof_constrains_ub, indpt_ub),
                "active_constrains" : active_val_constrains_type}
        return sols
      else:
        return f"No optimal solution founded: {res.message}"

    if case_2 and not case_3:
      res = linprog(cof_goal,  # Coeficientes de la funcion objetivo
                    A_eq= cof_constrains_eq, # Coeficientes de las restricciones
                    b_eq= indpt_eq, # Terminos independientes de las restricciones
                    bounds = bounds, # Limites
                    method = "highs") # Metodo recomendado en lugar de simplex
      if res.success:
        val_constrains = [sum(a*x for a, x in zip(constrain, res.x)) for constrain in cof_constrains_eq]
        clrncs = [abs(indpt_eq[i]-val_constrains[i]) for i in range(0,len(indpt_eq))]
        active_val_constrains_type = [i == 0 for i in clrncs]
        sols = {"vars_sols"         : np.array(list(res.x)),
                "fun_obj"           : res.fun,
                "slacks"            : clrncs,
                "sens"              : Transport_Models.get_sens_anyls(cof_goal, cof_constrains_eq, indpt_eq),
                "active_constrains" : active_val_constrains_type}
        return sols
      else:
        return f"No optimal solution founded: {res.message}"

    if case_3:
      res = linprog(cof_goal,  # Coeficientes de la funcion objetivo
                    A_ub = cof_constrains_ub,
                    b_ub = indpt_ub,
                    A_eq= cof_constrains_eq, # Coeficientes de las restricciones
                    b_eq= indpt_eq, # Terminos independientes de las restricciones
                    bounds = bounds, # Limites
                    method = "highs") # Metodo recomendado en lugar de simplex

      if res.success:
        cof_constrains = cof_constrains_ub + cof_constrains_eq
        indpt = indpt_eq + indpt_ub
        val_constrains = [sum(a*x for a, x in zip(constrain, res.x)) for constrain in cof_constrains]
        clrncs = [abs(indpt[i]-val_constrains[i]) for i in range(0,len(indpt))]
        active_val_constrains_type = [i == 0 for i in clrncs]
        sols = {"vars_sols"         : np.array(list(res.x)),
                "fun_obj"           : res.fun,
                "slacks"            : clrncs,
                "sens"              : Transport_Models.get_sens_anyls(cof_goal, cof_constrains, indpt),
                "active_constrains" : active_val_constrains_type}
        return sols
      else:
        return f"No optimal solution founded: {res.message}"

  @staticmethod
  def get_sens_anyls(cof_goal, cof_constrains, indpt):
    cof_goal_dual = np.array(indpt)  # Coeficientes duales son los límites originales
    cof_constrains_dual = np.array(cof_constrains).T  # Transposición para dual
    cof_constrains_dual_neg = -cof_constrains_dual  # Negar restricciones para "menor o igual"
    indpt_dual = -np.array(cof_goal)  # Límite independiente dual  print(cof_goal_dual)
    res_dual = linprog(cof_goal_dual,
                      A_ub= cof_constrains_dual_neg,
                      b_ub= indpt_dual,
                      method = "highs")
    val_constrains = [sum(a*x for a, x in zip(constrain, res_dual.x)) for constrain in cof_constrains_dual]
    clrncs = [abs(indpt[i]-val_constrains[i]) for i in range(0,len(indpt))]
    active_val_constrains_type = [i == 0 for i in clrncs]
    return {"shade_price"       : res_dual.x,
            "fun_obj"           : res_dual.fun,
            "slacks"            : clrncs,
            "active_constrains" : active_val_constrains_type}

  def __init__(self, costs, supply, demand):
    self.costs = np.array(costs.copy())
    self.supply = supply.copy()
    self.demand = demand.copy()

  def balance_model(self):
    total_supply = sum(self.supply)
    total_demand = sum(self.demand)

    if total_supply > total_demand:
      # Adding a fictial column of demand
      self.demand.append(total_supply - total_demand)
      for row in self.costs: # adding a column of zeros
        row.append(0)
    elif total_demand > total_supply:
      # Adding a ficticial row of supply
      self.supply.append(total_demand - total_supply)
      self.costs.append([0] * len(self.demand))

    return self.costs.copy(), self.supply.copy(), self.demand.copy()

  def total_costs(self, sol_asign):
    total = 0
    for i, row_sol in enumerate(sol_asign):
        total += sum([assign * cost for assign, cost in zip(row_sol, self.costs[i])])
    return total

  def general_method(self, bounds = None):
    n_rows, n_cols= self.costs.shape #
    bounds = bounds if bounds is not None else [(0, None)] * (n_rows * n_cols) #
    cof_goal = self.costs.flatten() #

    supply_matrix_cof = np.zeros((n_rows, n_rows * n_cols)) #
    for i in range(n_rows): #
      supply_matrix_cof[i, i * n_cols: (i + 1) * n_cols ] = 1 #

    demand_matrix_cof = np.zeros((n_cols, n_rows * n_cols))
    for j in range(n_cols):
      demand_matrix_cof[j, j:: n_cols] = 1

    cof_constrains = np.vstack([supply_matrix_cof, demand_matrix_cof]) # apilar
    indpt = np.concatenate([self.supply, self.demand]) # unir
    res = Transport_Models.get_complete_sol("Min",
                                            cof_goal,
                                            cof_constrains_eq = cof_constrains,
                                            indpt_eq = indpt)
    sol = res["vars_sols"].reshape(n_rows, n_cols)
    return {"allocation_matrix": sol, "total_cost": self.total_costs(sol), "method" : "simplex_method"}

  def northwest_corner(self):
    # Balancing the model
    costs, supply, demand = self.balance_model()

    rows = len(supply)
    cols = len(demand)
    sol = np.zeros((rows, cols))
    i = 0
    j = 0

    while i < rows and j < cols:
      asignated_val = min(supply[i], demand[j])
      sol[i][j] = asignated_val
      supply[i] -= asignated_val
      demand[j] -= asignated_val

      if supply[i] == 0:
        i += 1
      elif demand[j] == 0:
        j += 1

    return {"allocation_matrix": sol, "total_cost": self.total_costs(sol), "method" : "northwest_corner"}

  def min_cost(self):
    costs, supply, demand = self.balance_model()

    rows = len(supply)
    cols = len(demand)
    sol = np.zeros((rows, cols))

    # Getting objects and sorting
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
    rows = len(supply)
    cols = len(demand)
    sol = np.zeros((rows, cols))

    avaiable_cols = [_ for _ in range(cols)]
    avaiable_rows = [_ for _ in range(rows)]

    while len(avaiable_cols) != 0 and len(avaiable_rows) !=0:
      list_penalties = []
      for col in avaiable_cols:
        aux_list = []
        for row in avaiable_rows:
          aux_list.append(costs[row][col])
        hp.heapify(aux_list)
        first_min = hp.heappop(aux_list)
        second_min = hp.heappop(aux_list) if len(aux_list) !=0 else 2 * first_min
        # it's supposed that first_min < second_min
        list_penalties.append(self.Penaltie(second_min - first_min, col))
      selected_col = max(list_penalties).col
      posible_vals = []
      for row in avaiable_rows:
        posible_vals.append(self.Cell(costs[row][selected_col], row = row))
      selected_row = min(posible_vals).row

      asignated_val = min(supply[selected_row],demand[selected_col])
      sol[selected_row][selected_col] = asignated_val
      supply[selected_row] -= asignated_val
      demand[selected_col] -= asignated_val

      if supply[selected_row] > demand[selected_col]:
        avaiable_cols.remove(selected_col)
      elif demand[selected_col] > supply[selected_row]:
        avaiable_rows.remove(selected_row)
      elif supply[selected_row] == demand[selected_col]:
        avaiable_cols.remove(selected_col)
        avaiable_rows.remove(selected_row)

    return {"allocation_matrix": sol, "total_cost": self.total_costs(sol), "method": "vogel"}
  # Función para manejar el cálculo y mostrar los resultados
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
# Función para manejar el cálculo y mostrar los resultados
def solve_transport_problem():
    try:
        # Leer datos de la interfaz (como está actualmente)
        costs = cost_entry.get("1.0", "end-1c").strip()
        supply = supply_entry.get("1.0", "end-1c").strip()
        demand = demand_entry.get("1.0", "end-1c").strip()

        # Validar que todos los campos estén completos
        if not costs or not supply or not demand:
            messagebox.showwarning("Datos incompletos", "Debe rellenar los datos.")
            return

        # Convertir las entradas a listas y matrices
        costs = [[float(num) for num in line.split()] for line in costs.split("\n") if line.strip()]
        supply = [float(num) for num in supply.split() if num.strip()]
        demand = [float(num) for num in demand.split() if num.strip()]

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
    except Exception as e:
        messagebox.showerror("Error", f"Se produjo un error: {e}")

# Función para actualizar el ComboBox
def update_method_options():
    method_var.set("Método General")  # Establecer "Método General" como opción seleccionada
    methods_combobox["values"] = ["Esquina Noroeste", "Costo Mínimo", "Vogel", "Método General"]

# Crear la interfaz gráfica
root = tk.Tk()
root.title("Problema de Transporte")

# Crear los widgets de la interfaz
cost_label = tk.Label(root, text="Matriz de Costos (separada por espacios):")
cost_label.pack()

cost_entry = tk.Text(root, height=5, width=30)
cost_entry.pack()

supply_label = tk.Label(root, text="Oferta (separada por espacios):")
supply_label.pack()

supply_entry = tk.Text(root, height=2, width=30)
supply_entry.pack()

demand_label = tk.Label(root, text="Demanda (separada por espacios):")
demand_label.pack()

demand_entry = tk.Text(root, height=2, width=30)
demand_entry.pack()

method_label = tk.Label(root, text="Selecciona el Método:")
method_label.pack()

method_var = tk.StringVar()
methods_combobox = ttk.Combobox(root, textvariable=method_var)
methods_combobox["values"] = ["Esquina Noroeste", "Costo Mínimo", "Vogel"]
methods_combobox.pack()

solve_button = tk.Button(root, text="Resolver", command=solve_transport_problem)
solve_button.pack()

output_text = tk.StringVar()
output_label = tk.Label(root, textvariable=output_text)
output_label.pack()

update_method_options()  # Llamar para incluir "Método General"

# Ejecutar la interfaz gráfica
root.mainloop()