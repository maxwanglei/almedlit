import { Spinner } from "@astryxdesign/core/Spinner";

export default function LoadingState({ label }: { label: string }): React.ReactElement {
  return <Spinner className="loading-state" label={label} size="md" />;
}
