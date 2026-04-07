defmodule SrfmCoordination.ParamValidator do
  @moduledoc """
  Extended parameter validation for SRFM trading system parameters.

  Validates individual fields against domain bounds and cross-parameter
  constraints including stationarity conditions, GARCH risk rules, and
  time-window consistency checks.

  Individual field rules:
    BH_MASS_THRESH      in [0.5, 5.0]
    NAV_OMEGA_SCALE_K   in [0.1, 2.0]
    NAV_GEO_ENTRY_GATE  in [0.5, 10.0]
    GARCH_ALPHA + GARCH_BETA < 1.0  (stationarity)
    MIN_HOLD_BARS       integer in [1, 96]
    BLOCKED_HOURS       subset of 0..23
    HURST_WINDOW        in [50, 500]
    CF_LONG > CF_SHORT > 0

  Cross-parameter constraints:
    If GARCH_ALPHA > 0.3 then RISK_MULTIPLIER <= 1.0
    MIN_HOLD_BARS * 15 minutes <= 24 hours (MIN_HOLD_BARS <= 96)
  """

  require Logger

  # ---------------------------------------------------------------------------
  # Types
  # ---------------------------------------------------------------------------

  @type param_map :: %{String.t() => term()}

  @type validation_error :: %{
          field: String.t(),
          rule: atom(),
          message: String.t(),
          value: term()
        }

  @type constraint_violation :: %{
          fields: [String.t()],
          rule: atom(),
          message: String.t(),
          old_values: map(),
          new_values: map()
        }

  # ---------------------------------------------------------------------------
  # Public API
  # ---------------------------------------------------------------------------

  @doc """
  Validate a parameter map against all field rules and cross-parameter constraints.

  Returns :ok if all rules pass, or {:error, [validation_error]} with a list
  of all violations found (not short-circuiting).
  """
  @spec validate(param_map()) :: :ok | {:error, [validation_error()]}
  def validate(params) when is_map(params) do
    errors =
      []
      |> validate_bh_mass_thresh(params)
      |> validate_nav_omega_scale_k(params)
      |> validate_nav_geo_entry_gate(params)
      |> validate_garch_stationarity(params)
      |> validate_min_hold_bars(params)
      |> validate_blocked_hours(params)
      |> validate_hurst_window(params)
      |> validate_cf_long_short(params)
      |> validate_garch_risk_multiplier(params)
      |> validate_min_hold_bars_24h(params)

    case errors do
      [] -> :ok
      _ -> {:error, Enum.reverse(errors)}
    end
  end

  @doc """
  Check cross-parameter constraints when transitioning from old_params to new_params.

  Returns a (possibly empty) list of ConstraintViolation structs describing
  any constraints broken by the transition.
  """
  @spec constraint_violations(param_map(), param_map()) :: [constraint_violation()]
  def constraint_violations(old_params, new_params) when is_map(old_params) and is_map(new_params) do
    merged = Map.merge(old_params, new_params)

    []
    |> check_garch_risk_transition(old_params, new_params, merged)
    |> check_hold_bars_transition(old_params, new_params, merged)
    |> check_cf_direction_preserved(old_params, new_params, merged)
    |> check_garch_stationarity_transition(old_params, new_params, merged)
  end

  @doc "Validate a single field value against its rule."
  @spec validate_field(String.t(), term()) :: :ok | {:error, validation_error()}
  def validate_field("BH_MASS_THRESH", value) do
    check_range("BH_MASS_THRESH", value, 0.5, 5.0, :range_check)
  end

  def validate_field("NAV_OMEGA_SCALE_K", value) do
    check_range("NAV_OMEGA_SCALE_K", value, 0.1, 2.0, :range_check)
  end

  def validate_field("NAV_GEO_ENTRY_GATE", value) do
    check_range("NAV_GEO_ENTRY_GATE", value, 0.5, 10.0, :range_check)
  end

  def validate_field("MIN_HOLD_BARS", value) do
    with :ok <- check_integer("MIN_HOLD_BARS", value),
         :ok <- check_range("MIN_HOLD_BARS", value, 1, 96, :range_check) do
      :ok
    end
  end

  def validate_field("HURST_WINDOW", value) do
    with :ok <- check_integer("HURST_WINDOW", value),
         :ok <- check_range("HURST_WINDOW", value, 50, 500, :range_check) do
      :ok
    end
  end

  def validate_field("BLOCKED_HOURS", value) do
    validate_blocked_hours_value("BLOCKED_HOURS", value)
  end

  def validate_field(_field, _value), do: :ok

  # ---------------------------------------------------------------------------
  # Individual field validators (accumulate into error list)
  # ---------------------------------------------------------------------------

  defp validate_bh_mass_thresh(errors, params) do
    case Map.fetch(params, "BH_MASS_THRESH") do
      {:ok, val} ->
        case check_range("BH_MASS_THRESH", val, 0.5, 5.0, :range_check) do
          :ok -> errors
          {:error, e} -> [e | errors]
        end

      :error ->
        errors
    end
  end

  defp validate_nav_omega_scale_k(errors, params) do
    case Map.fetch(params, "NAV_OMEGA_SCALE_K") do
      {:ok, val} ->
        case check_range("NAV_OMEGA_SCALE_K", val, 0.1, 2.0, :range_check) do
          :ok -> errors
          {:error, e} -> [e | errors]
        end

      :error ->
        errors
    end
  end

  defp validate_nav_geo_entry_gate(errors, params) do
    case Map.fetch(params, "NAV_GEO_ENTRY_GATE") do
      {:ok, val} ->
        case check_range("NAV_GEO_ENTRY_GATE", val, 0.5, 10.0, :range_check) do
          :ok -> errors
          {:error, e} -> [e | errors]
        end

      :error ->
        errors
    end
  end

  defp validate_garch_stationarity(errors, params) do
    alpha = Map.get(params, "GARCH_ALPHA")
    beta = Map.get(params, "GARCH_BETA")

    cond do
      is_nil(alpha) or is_nil(beta) ->
        errors

      not is_number(alpha) ->
        [
          make_error("GARCH_ALPHA", :type_check, "must be a number", alpha)
          | errors
        ]

      not is_number(beta) ->
        [
          make_error("GARCH_BETA", :type_check, "must be a number", beta)
          | errors
        ]

      alpha + beta >= 1.0 ->
        [
          make_error(
            "GARCH_ALPHA+GARCH_BETA",
            :garch_stationarity,
            "GARCH_ALPHA + GARCH_BETA must be < 1.0 for stationarity (got #{alpha + beta})",
            %{alpha: alpha, beta: beta, sum: alpha + beta}
          )
          | errors
        ]

      true ->
        errors
    end
  end

  defp validate_min_hold_bars(errors, params) do
    case Map.fetch(params, "MIN_HOLD_BARS") do
      {:ok, val} ->
        errs =
          []
          |> then(fn acc ->
            case check_integer("MIN_HOLD_BARS", val) do
              :ok -> acc
              {:error, e} -> [e | acc]
            end
          end)
          |> then(fn acc ->
            case check_range("MIN_HOLD_BARS", val, 1, 96, :range_check) do
              :ok -> acc
              {:error, e} -> [e | acc]
            end
          end)

        errs ++ errors

      :error ->
        errors
    end
  end

  defp validate_blocked_hours(errors, params) do
    case Map.fetch(params, "BLOCKED_HOURS") do
      {:ok, val} ->
        case validate_blocked_hours_value("BLOCKED_HOURS", val) do
          :ok -> errors
          {:error, e} -> [e | errors]
        end

      :error ->
        errors
    end
  end

  defp validate_blocked_hours_value(field, val) do
    cond do
      not is_list(val) ->
        {:error, make_error(field, :type_check, "must be a list of integers 0-23", val)}

      not Enum.all?(val, &(is_integer(&1) and &1 >= 0 and &1 <= 23)) ->
        bad = Enum.reject(val, &(is_integer(&1) and &1 >= 0 and &1 <= 23))

        {:error,
         make_error(
           field,
           :range_check,
           "all hours must be integers in 0..23, invalid: #{inspect(bad)}",
           val
         )}

      length(val) != length(Enum.uniq(val)) ->
        {:error, make_error(field, :uniqueness, "hours must be unique", val)}

      true ->
        :ok
    end
  end

  defp validate_hurst_window(errors, params) do
    case Map.fetch(params, "HURST_WINDOW") do
      {:ok, val} ->
        errs =
          []
          |> then(fn acc ->
            case check_integer("HURST_WINDOW", val) do
              :ok -> acc
              {:error, e} -> [e | acc]
            end
          end)
          |> then(fn acc ->
            case check_range("HURST_WINDOW", val, 50, 500, :range_check) do
              :ok -> acc
              {:error, e} -> [e | acc]
            end
          end)

        errs ++ errors

      :error ->
        errors
    end
  end

  defp validate_cf_long_short(errors, params) do
    cf_long = Map.get(params, "CF_LONG")
    cf_short = Map.get(params, "CF_SHORT")

    cond do
      is_nil(cf_long) and is_nil(cf_short) ->
        errors

      is_nil(cf_long) or is_nil(cf_short) ->
        errors

      not is_number(cf_long) ->
        [make_error("CF_LONG", :type_check, "must be a number", cf_long) | errors]

      not is_number(cf_short) ->
        [make_error("CF_SHORT", :type_check, "must be a number", cf_short) | errors]

      cf_short <= 0 ->
        [
          make_error("CF_SHORT", :cf_positive, "CF_SHORT must be > 0 (got #{cf_short})", cf_short)
          | errors
        ]

      cf_long <= cf_short ->
        [
          make_error(
            "CF_LONG",
            :cf_ordering,
            "CF_LONG must be > CF_SHORT (CF_LONG=#{cf_long}, CF_SHORT=#{cf_short})",
            %{cf_long: cf_long, cf_short: cf_short}
          )
          | errors
        ]

      true ->
        errors
    end
  end

  # Cross-param: if GARCH_ALPHA > 0.3 then RISK_MULTIPLIER must be <= 1.0
  defp validate_garch_risk_multiplier(errors, params) do
    alpha = Map.get(params, "GARCH_ALPHA")
    risk_mult = Map.get(params, "RISK_MULTIPLIER")

    if is_number(alpha) and alpha > 0.3 and is_number(risk_mult) and risk_mult > 1.0 do
      [
        make_error(
          "RISK_MULTIPLIER",
          :garch_risk_constraint,
          "When GARCH_ALPHA > 0.3 (#{alpha}), RISK_MULTIPLIER must be <= 1.0 (got #{risk_mult})",
          %{garch_alpha: alpha, risk_multiplier: risk_mult}
        )
        | errors
      ]
    else
      errors
    end
  end

  # Cross-param: MIN_HOLD_BARS * 15 minutes <= 24 hours (i.e. <= 96)
  defp validate_min_hold_bars_24h(errors, params) do
    case Map.fetch(params, "MIN_HOLD_BARS") do
      {:ok, val} when is_integer(val) and val > 96 ->
        [
          make_error(
            "MIN_HOLD_BARS",
            :hold_bars_24h,
            "MIN_HOLD_BARS * 15 min must be <= 24 hours (max 96 bars, got #{val})",
            val
          )
          | errors
        ]

      _ ->
        errors
    end
  end

  # ---------------------------------------------------------------------------
  # Cross-parameter constraint checks (for constraint_violations/2)
  # ---------------------------------------------------------------------------

  defp check_garch_risk_transition(violations, old_params, new_params, merged) do
    old_alpha = Map.get(old_params, "GARCH_ALPHA")
    new_alpha = Map.get(new_params, "GARCH_ALPHA")
    new_risk = Map.get(new_params, "RISK_MULTIPLIER") || Map.get(merged, "RISK_MULTIPLIER")
    effective_alpha = new_alpha || old_alpha

    if is_number(effective_alpha) and effective_alpha > 0.3 and
         is_number(new_risk) and new_risk > 1.0 do
      [
        %{
          fields: ["GARCH_ALPHA", "RISK_MULTIPLIER"],
          rule: :garch_risk_constraint,
          message:
            "GARCH_ALPHA=#{effective_alpha} > 0.3 requires RISK_MULTIPLIER <= 1.0, " <>
              "but new value is #{new_risk}",
          old_values: Map.take(old_params, ["GARCH_ALPHA", "RISK_MULTIPLIER"]),
          new_values: Map.take(new_params, ["GARCH_ALPHA", "RISK_MULTIPLIER"])
        }
        | violations
      ]
    else
      violations
    end
  end

  defp check_hold_bars_transition(violations, old_params, new_params, _merged) do
    new_bars = Map.get(new_params, "MIN_HOLD_BARS") || Map.get(old_params, "MIN_HOLD_BARS")

    if is_integer(new_bars) and new_bars > 96 do
      [
        %{
          fields: ["MIN_HOLD_BARS"],
          rule: :hold_bars_24h,
          message: "MIN_HOLD_BARS=#{new_bars} exceeds 96 (24-hour limit at 15min bars)",
          old_values: Map.take(old_params, ["MIN_HOLD_BARS"]),
          new_values: Map.take(new_params, ["MIN_HOLD_BARS"])
        }
        | violations
      ]
    else
      violations
    end
  end

  defp check_cf_direction_preserved(violations, old_params, new_params, merged) do
    cf_long = Map.get(new_params, "CF_LONG") || Map.get(merged, "CF_LONG")
    cf_short = Map.get(new_params, "CF_SHORT") || Map.get(merged, "CF_SHORT")

    cond do
      is_nil(cf_long) or is_nil(cf_short) ->
        violations

      is_number(cf_long) and is_number(cf_short) and cf_long <= cf_short ->
        [
          %{
            fields: ["CF_LONG", "CF_SHORT"],
            rule: :cf_ordering,
            message: "CF_LONG (#{cf_long}) must be > CF_SHORT (#{cf_short})",
            old_values: Map.take(old_params, ["CF_LONG", "CF_SHORT"]),
            new_values: Map.take(new_params, ["CF_LONG", "CF_SHORT"])
          }
          | violations
        ]

      true ->
        violations
    end
  end

  defp check_garch_stationarity_transition(violations, old_params, new_params, merged) do
    alpha = Map.get(new_params, "GARCH_ALPHA") || Map.get(merged, "GARCH_ALPHA")
    beta = Map.get(new_params, "GARCH_BETA") || Map.get(merged, "GARCH_BETA")

    if is_number(alpha) and is_number(beta) and alpha + beta >= 1.0 do
      [
        %{
          fields: ["GARCH_ALPHA", "GARCH_BETA"],
          rule: :garch_stationarity,
          message:
            "GARCH_ALPHA + GARCH_BETA = #{alpha + beta} >= 1.0 violates stationarity constraint",
          old_values: Map.take(old_params, ["GARCH_ALPHA", "GARCH_BETA"]),
          new_values: Map.take(new_params, ["GARCH_ALPHA", "GARCH_BETA"])
        }
        | violations
      ]
    else
      violations
    end
  end

  # ---------------------------------------------------------------------------
  # Field-level check helpers
  # ---------------------------------------------------------------------------

  defp check_range(field, value, lo, hi, rule) when is_number(value) do
    if value >= lo and value <= hi do
      :ok
    else
      {:error,
       make_error(
         field,
         rule,
         "#{field} must be in [#{lo}, #{hi}] (got #{value})",
         value
       )}
    end
  end

  defp check_range(field, value, _lo, _hi, _rule) do
    {:error, make_error(field, :type_check, "#{field} must be a number (got #{inspect(value)})", value)}
  end

  defp check_integer(field, value) when is_integer(value), do: :ok

  defp check_integer(field, value) do
    {:error,
     make_error(field, :type_check, "#{field} must be an integer (got #{inspect(value)})", value)}
  end

  defp make_error(field, rule, message, value) do
    %{field: field, rule: rule, message: message, value: value}
  end
end
