import { CartesianGrid, Line, LineChart, ReferenceDot, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { compactMoney, money } from "../../lib/format";
import type { CurvePoint } from "../../lib/types";

interface Props {
  curve: CurvePoint[];
  current: CurvePoint;
  recommended: CurvePoint;
}

export function NetCurve({ curve, current, recommended }: Props) {
  const data = curve.map((p) => ({ t: Math.round(p.threshold * 100), net: p.net }));
  return (
    <figure>
      <figcaption className="text-sm text-ink-muted">
        Net value by risk cutoff on your history · <span className="text-gain">●</span> recommended ·{" "}
        <span className="text-verdict">●</span> current
      </figcaption>
      <div className="mt-2 h-48">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: 4 }}>
            <CartesianGrid stroke="#e3dccf" vertical={false} />
            <XAxis
              dataKey="t"
              type="number"
              domain={[0, 100]}
              ticks={[0, 25, 50, 75, 100]}
              tickFormatter={(v) => `${v}%`}
              stroke="#736b5f"
              fontSize={12}
            />
            <YAxis tickFormatter={(v) => compactMoney(Number(v))} stroke="#736b5f" fontSize={12} width={56} />
            <Tooltip formatter={(v) => [money(Number(v)), "Net"]} labelFormatter={(v) => `Cutoff ${v}%`} />
            <ReferenceLine y={0} stroke="#736b5f" />
            <Line type="monotone" dataKey="net" stroke="#1c1a17" dot={false} strokeWidth={1.5} isAnimationActive={false} />
            <ReferenceDot x={Math.round(recommended.threshold * 100)} y={recommended.net} r={4} fill="#2f6b3a" stroke="none" />
            <ReferenceDot x={Math.round(current.threshold * 100)} y={current.net} r={5} fill="#a4262c" stroke="#fbf9f4" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </figure>
  );
}
