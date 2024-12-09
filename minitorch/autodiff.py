from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Protocol


# ## Task 1.1

# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    vals_list = list(vals).copy()

    vals_list_plus = vals_list.copy()
    vals_list_plus[arg] += epsilon
    f_plus = f(*vals_list_plus)

    vals_list_minus = vals_list.copy()
    vals_list_minus[arg] -= epsilon
    f_minus = f(*vals_list_minus)

    return (f_plus - f_minus) / (2.0 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulates the derivative for this variable.

        This variable should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x: The value to be added to the current derivative.

        """
        ...

    @property
    def unique_id(self) -> int:
        """Returns the unique ID of this variable."""
        ...

    def is_leaf(self) -> bool:
        """Returns True if this variable is a leaf."""
        ...

    def is_constant(self) -> bool:
        """Returns True if this variable is constant."""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns the parent variables of this variable."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Computes the chain rule for this variable.

        Args:
        ----
            d_output: The derivative of the output with respect to this variable.

        Returns:
        -------
            An iterable of tuples, where each tuple contains a parent variable and its derivative.

        """
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    # ASSIGNMENT 1.4

    order: List[Variable] = []
    seen = set()

    def visit(var: Variable) -> None:
        if var.unique_id in seen or var.is_constant():
            return
        if not var.is_leaf():
            for parent in var.parents:
                if not parent.is_constant():
                    visit(parent)
        seen.add(var.unique_id)
        order.insert(0, var)

    visit(variable)
    return order

    # END ASSIGNMENT 1.4


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order tocompute derivatives for the leave nodes.

    Args:
    ----
        variable: The right-most variable
        deriv: Its derivative that we want to propagate backward to the leaves.

    """
    sorted_variables = topological_sort(variable)
    derivatives = {}
    derivatives[variable.unique_id] = deriv

    for var in sorted_variables:
        deriv = derivatives[var.unique_id]

        if var.is_leaf():
            var.accumulate_derivative(deriv)
        else:
            for parent, parent_deriv in var.chain_rule(deriv):
                if parent.is_constant():
                    continue
                derivatives.setdefault(parent.unique_id, 0)
                derivatives[parent.unique_id] += parent_deriv


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the saved values from the forward method."""
        return self.saved_values
